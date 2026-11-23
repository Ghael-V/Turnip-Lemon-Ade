/*
 * Copyright © 2015 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/* A collection of unit tests for cache.c */

#include <gtest/gtest.h>

#include <string>

#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <ftw.h>
#include <errno.h>
#include <stdarg.h>
#include <inttypes.h>
#include <limits.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <utime.h>

#include "util/detect_os.h"
#include "util/disk_cache_os.h"
#include "util/disk_cache.h"
#include "util/macros.h"
#include "util/mesa-blake3.h"
#include "util/os_misc.h"
#include "util/ralloc.h"
#include "util/u_atomic.h"

#ifdef FOZ_DB_UTIL_DYNAMIC_LIST
#include <sys/inotify.h>
#endif

#ifdef ENABLE_SHADER_CACHE

/* Callback for nftw used in rmrf_local below.
 */
static int
remove_entry(const char *path,
             const struct stat *sb,
             int typeflag,
             struct FTW *ftwbuf)
{
   int err = remove(path);

   if (err)
      fprintf(stderr, "Error removing %s: %s\n", path, strerror(errno));

   return err;
}

/* Recursively remove a directory.
 *
 * This is equivalent to "rm -rf <dir>" with one bit of protection
 * that the directory name must begin with "." to ensure we don't
 * wander around deleting more than intended.
 *
 * Returns 0 on success, -1 on any error.
 */
static int
rmrf_local(const char *path)
{
   if (path == NULL || *path == '\0' || *path != '.')
      return -1;

   return nftw(path, remove_entry, 64, FTW_DEPTH | FTW_PHYS);
}

static void
check_directories_created(void *mem_ctx, const char *cache_dir)
{
   bool sub_dirs_created = false;

   char buf[PATH_MAX];
   if (getcwd(buf, PATH_MAX)) {
      char *full_path = ralloc_asprintf(mem_ctx, "%s%s", buf, ++cache_dir);
      struct stat sb;
      if (stat(full_path, &sb) != -1 && S_ISDIR(sb.st_mode))
         sub_dirs_created = true;
   }

   EXPECT_TRUE(sub_dirs_created) << "create sub dirs";
}

static bool
does_cache_contain(struct disk_cache *cache, const cache_key key)
{
   void *result;

   result = disk_cache_get(cache, key, NULL);

   if (result) {
      free(result);
      return true;
   }

   return false;
}

static bool
cache_exists(struct disk_cache *cache)
{
   uint8_t key[BLAKE3_KEY_LEN];
   char data[] = "some test data";

   if (!cache)
      return false;

   disk_cache_compute_key(cache, data, sizeof(data), key);
   disk_cache_put(cache, key, data, sizeof(data), NULL);
   disk_cache_wait_for_idle(cache);
   void *result = disk_cache_get(cache, key, NULL);
   disk_cache_remove(cache, key);

   free(result);
   return result != NULL;
}

#ifdef FOZ_DB_UTIL_DYNAMIC_LIST
/*
 * Wait for foz_db updater to close the foz db list file after its
 * done updating.
 */
static void
close_and_wait_for_list(FILE* list_file, const char* list_filename)
{
   char buf[10 * (sizeof(struct inotify_event) + NAME_MAX + 1)];
   int fd = inotify_init1(IN_CLOEXEC);
   int wd = inotify_add_watch(fd, list_filename, IN_CLOSE_NOWRITE);
   if (wd < 0) {
       close(fd);
       return;
   }

   /* Prevent racing by closing the list file we wrote to after inotify
    * has started. Since this was a write, it gets ignored by inotify
    */
   fclose(list_file);

   int len = read(fd, buf, sizeof(buf));
   if (len == -1)  {
      close(fd);
      inotify_rm_watch(fd, wd);
      return;
   }

   int i = 0;
   while (i < len) {
      struct inotify_event *event = (struct inotify_event *)&buf[i];
      i += sizeof(struct inotify_event) + event->len;
      if (event->mask & IN_CLOSE_NOWRITE)
         break;
   }

   inotify_rm_watch(fd, wd);
   close(fd);
}
#endif

#define CACHE_TEST_TMP "./cache-test-tmp"

static void
test_disk_cache_create(void *mem_ctx, const char *cache_dir_name,
                       const char *driver_id)
{
   struct disk_cache *cache;
   int err;

   /* Before doing anything else, ensure that with
    * MESA_SHADER_CACHE_DISABLE set to true, that disk_cache_create returns NO-OP cache.
    */
   os_set_option("MESA_SHADER_CACHE_DISABLE", "true", true);
   cache = disk_cache_create("test", driver_id, 0);
   EXPECT_EQ(cache->type, DISK_CACHE_NONE) << "disk_cache_create with MESA_SHADER_CACHE_DISABLE set";
   disk_cache_destroy(cache);

   os_unset_option("MESA_SHADER_CACHE_DISABLE");

#ifdef SHADER_CACHE_DISABLE_BY_DEFAULT
   /* With SHADER_CACHE_DISABLE_BY_DEFAULT, ensure that with
    * MESA_SHADER_CACHE_DISABLE set to nothing, disk_cache_create returns NO-OP cache.
    */
   os_unset_option("MESA_SHADER_CACHE_DISABLE");
   cache = disk_cache_create("test", driver_id, 0);
   EXPECT_EQ(cache->type, DISK_CACHE_NONE)
      << "disk_cache_create with MESA_SHADER_CACHE_DISABLE unset "
         "and SHADER_CACHE_DISABLE_BY_DEFAULT build option";
   disk_cache_destroy(cache);

   /* For remaining tests, ensure that the cache is enabled. */
   os_set_option("MESA_SHADER_CACHE_DISABLE", "false", true);
#endif /* SHADER_CACHE_DISABLE_BY_DEFAULT */

   /* For the first real disk_cache_create() clear these environment
    * variables to test creation of cache in home directory.
    */
   os_unset_option("MESA_SHADER_CACHE_DIR");
   os_unset_option("XDG_CACHE_HOME");

   cache = disk_cache_create("test", driver_id, 0);
   EXPECT_NE(cache, nullptr) << "disk_cache_create with no environment variables";

   disk_cache_destroy(cache);

#if DETECT_OS_ANDROID
   /* Android doesn't try writing to disk (just calls the cache callbacks), so
    * the directory tests below don't apply.
    */
   return;
#endif

   /* Test with XDG_CACHE_HOME set */
   os_set_option("XDG_CACHE_HOME", CACHE_TEST_TMP "/xdg-cache-home", true);
   cache = disk_cache_create("test", driver_id, 0);
   EXPECT_TRUE(cache_exists(cache))
      << "disk_cache_create with XDG_CACHE_HOME set with a non-existing parent directory";

   char *path = ralloc_asprintf(
      mem_ctx, "%s%s", CACHE_TEST_TMP "/xdg-cache-home/", cache_dir_name);
   check_directories_created(mem_ctx, path);

   disk_cache_destroy(cache);

   /* Test with MESA_SHADER_CACHE_DIR set */
   err = rmrf_local(CACHE_TEST_TMP);
   EXPECT_EQ(err, 0) << "Removing " CACHE_TEST_TMP;

   os_set_option("MESA_SHADER_CACHE_DIR", CACHE_TEST_TMP "/mesa-shader-cache-dir", true);
   cache = disk_cache_create("test", driver_id, 0);
   EXPECT_TRUE(cache_exists(cache))
      << "disk_cache_create with MESA_SHADER_CACHE_DIR set with a non-existing parent directory";

   disk_cache_destroy(cache);
   rmrf_local(CACHE_TEST_TMP);
   EXPECT_EQ(err, 0) << "Removing " CACHE_TEST_TMP;

   err = mkdir(CACHE_TEST_TMP, 0755);
   if (err != 0) {
      fprintf(stderr, "Error creating %s: %s\n", CACHE_TEST_TMP, strerror(errno));
      GTEST_FAIL();
   }

   cache = disk_cache_create("test", driver_id, 0);
   EXPECT_TRUE(cache_exists(cache)) << "disk_cache_create with MESA_SHADER_CACHE_DIR set with existing parent directory";

   path = ralloc_asprintf(
      mem_ctx, "%s%s", CACHE_TEST_TMP "/mesa-shader-cache-dir/", cache_dir_name);
   check_directories_created(mem_ctx, path);

   disk_cache_destroy(cache);
}

static void
test_put_and_get(bool test_cache_size_limit, const char *driver_id)
{
   struct disk_cache *cache;
   char blob[] = "This is a blob of thirty-seven bytes";
   uint8_t blob_key[BLAKE3_KEY_LEN];
   char string[] = "While this string has thirty-four";
   uint8_t string_key[BLAKE3_KEY_LEN];
   char *result;
   size_t size;
   uint8_t *one_KB, *one_MB;
   uint8_t one_KB_key[BLAKE3_KEY_LEN], one_MB_key[BLAKE3_KEY_LEN];
   int count;

#ifdef SHADER_CACHE_DISABLE_BY_DEFAULT
   os_set_option("MESA_SHADER_CACHE_DISABLE", "false", true);
#endif /* SHADER_CACHE_DISABLE_BY_DEFAULT */

   cache = disk_cache_create("test", driver_id, 0);

   disk_cache_compute_key(cache, blob, sizeof(blob), blob_key);

   /* Ensure that disk_cache_get returns nothing before anything is added. */
   result = (char *) disk_cache_get(cache, blob_key, &size);
   EXPECT_EQ(result, nullptr) << "disk_cache_get with non-existent item (pointer)";
   EXPECT_EQ(size, 0) << "disk_cache_get with non-existent item (size)";

   /* Simple test of put and get. */
   disk_cache_put(cache, blob_key, blob, sizeof(blob), NULL);

   /* disk_cache_put() hands things off to a thread so wait for it. */
   disk_cache_wait_for_idle(cache);

   result = (char *) disk_cache_get(cache, blob_key, &size);
   EXPECT_STREQ(blob, result) << "disk_cache_get of existing item (pointer)";
   EXPECT_EQ(size, sizeof(blob)) << "disk_cache_get of existing item (size)";

   free(result);

   /* Test put and get of a second item. */
   disk_cache_compute_key(cache, string, sizeof(string), string_key);
   disk_cache_put(cache, string_key, string, sizeof(string), NULL);

   /* disk_cache_put() hands things off to a thread so wait for it. */
   disk_cache_wait_for_idle(cache);

   result = (char *) disk_cache_get(cache, string_key, &size);
   EXPECT_STREQ(result, string) << "2nd disk_cache_get of existing item (pointer)";
   EXPECT_EQ(size, sizeof(string)) << "2nd disk_cache_get of existing item (size)";

   free(result);

   /* Set the cache size to 1KB and add a 1KB item to force an eviction. */
   disk_cache_destroy(cache);

   if (!test_cache_size_limit)
      return;

   os_set_option("MESA_SHADER_CACHE_MAX_SIZE", "1K", true);
   cache = disk_cache_create("test", driver_id, 0);

   one_KB = (uint8_t *) calloc(1, 1024);

   /* Obviously the BLAKE3 hash of 1024 zero bytes isn't particularly
    * interesting. But we do have want to take some special care with
    * the hash we use here. The issue is that in this artificial case,
    * (with only three files in the cache), the probability is good
    * that each of the three files will end up in their own
    * directory. Then, if the directory containing the .tmp file for
    * the new item being added for disk_cache_put() is the chosen victim
    * directory for eviction, then no suitable file will be found and
    * nothing will be evicted.
    *
    * That's actually expected given how the eviction code is
    * implemented, (which expects to only evict once things are more
    * interestingly full than that).
    *
    * For this test, we force this signature to land in the same
    * directory as the original blob first written to the cache.
    */
   disk_cache_compute_key(cache, one_KB, 1024, one_KB_key);
   one_KB_key[0] = blob_key[0];

   disk_cache_put(cache, one_KB_key, one_KB, 1024, NULL);

   free(one_KB);

   /* disk_cache_put() hands things off to a thread so wait for it. */
   disk_cache_wait_for_idle(cache);

   result = (char *) disk_cache_get(cache, one_KB_key, &size);
   EXPECT_NE(result, nullptr) << "3rd disk_cache_get of existing item (pointer)";
   EXPECT_EQ(size, 1024) << "3rd disk_cache_get of existing item (size)";

   free(result);

   /* Ensure eviction happened by checking that both of the previous
    * cache itesm were evicted.
    */
   bool contains_1KB_file = false;
   count = 0;
   if (does_cache_contain(cache, blob_key))
       count++;

   if (does_cache_contain(cache, string_key))
       count++;

   if (does_cache_contain(cache, one_KB_key)) {
      count++;
      contains_1KB_file = true;
   }

   EXPECT_TRUE(contains_1KB_file)
      << "disk_cache_put eviction last file == MAX_SIZE (1KB)";
   EXPECT_EQ(count, 1) << "disk_cache_put eviction with MAX_SIZE=1K";

   /* Now increase the size to 1M, add back both items, and ensure all
    * three that have been added are available via disk_cache_get.
    */
   disk_cache_destroy(cache);

   os_set_option("MESA_SHADER_CACHE_MAX_SIZE", "1M", true);
   cache = disk_cache_create("test", driver_id, 0);

   disk_cache_put(cache, blob_key, blob, sizeof(blob), NULL);
   disk_cache_put(cache, string_key, string, sizeof(string), NULL);

   /* disk_cache_put() hands things off to a thread so wait for it. */
   disk_cache_wait_for_idle(cache);

   count = 0;
   if (does_cache_contain(cache, blob_key))
       count++;

   if (does_cache_contain(cache, string_key))
       count++;

   if (does_cache_contain(cache, one_KB_key))
       count++;

   EXPECT_EQ(count, 3) << "no eviction before overflow with MAX_SIZE=1M";

   /* Finally, check eviction again after adding an object of size 1M. */
   one_MB = (uint8_t *) calloc(1024, 1024);

   disk_cache_compute_key(cache, one_MB, 1024 * 1024, one_MB_key);
   one_MB_key[0] = blob_key[0];

   disk_cache_put(cache, one_MB_key, one_MB, 1024 * 1024, NULL);

   free(one_MB);

   /* disk_cache_put() hands things off to a thread so wait for it. */
   disk_cache_wait_for_idle(cache);

   bool contains_1MB_file = false;
   count = 0;
   if (does_cache_contain(cache, blob_key))
       count++;

   if (does_cache_contain(cache, string_key))
       count++;

   if (does_cache_contain(cache, one_KB_key))
       count++;

   if (does_cache_contain(cache, one_MB_key)) {
      count++;
      contains_1MB_file = true;
   }

   EXPECT_TRUE(contains_1MB_file)
      << "disk_cache_put eviction last file == MAX_SIZE (1MB)";
   EXPECT_EQ(count, 1) << "eviction after overflow with MAX_SIZE=1M";

   disk_cache_destroy(cache);
}

static void
test_put_key_and_get_key(const char *driver_id)
{
   struct disk_cache *cache;
   bool result;

   uint8_t key_a[BLAKE3_KEY_LEN] = {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                                         10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
   uint8_t key_b[BLAKE3_KEY_LEN] = { 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                                         30, 33, 32, 33, 34, 35, 36, 37, 38, 39};
   uint8_t key_a_collide[BLAKE3_KEY_LEN] =
                        { 0,  1, 42, 43, 44, 45, 46, 47, 48, 49,
                         50, 55, 52, 53, 54, 55, 56, 57, 58, 59};

#ifdef SHADER_CACHE_DISABLE_BY_DEFAULT
   os_set_option("MESA_SHADER_CACHE_DISABLE", "false", true);
#endif /* SHADER_CACHE_DISABLE_BY_DEFAULT */

   cache = disk_cache_create("test", driver_id, 0);

   /* First test that disk_cache_has_key returns false before disk_cache_put_key */
   result = disk_cache_has_key(cache, key_a);
   EXPECT_EQ(result, 0) << "disk_cache_has_key before key added";

   /* Then a couple of tests of disk_cache_put_key followed by disk_cache_has_key */
   disk_cache_put_key(cache, key_a);
   result = disk_cache_has_key(cache, key_a);
   EXPECT_EQ(result, 1) << "disk_cache_has_key after key added";

   disk_cache_put_key(cache, key_b);
   result = disk_cache_has_key(cache, key_b);
   EXPECT_EQ(result, 1) << "2nd disk_cache_has_key after key added";

   /* Test that a key with the same two bytes as an existing key
    * forces an eviction.
    */
   disk_cache_put_key(cache, key_a_collide);
   result = disk_cache_has_key(cache, key_a_collide);
   EXPECT_EQ(result, 1) << "put_key of a colliding key lands in the cache";

   result = disk_cache_has_key(cache, key_a);
   EXPECT_EQ(result, 0) << "put_key of a colliding key evicts from the cache";

   /* And finally test that we can re-add the original key to re-evict
    * the colliding key.
    */
   disk_cache_put_key(cache, key_a);
   result = disk_cache_has_key(cache, key_a);
   EXPECT_EQ(result, 1) << "put_key of original key lands again";

   result = disk_cache_has_key(cache, key_a_collide);
   EXPECT_EQ(result, 0) << "put_key of orginal key evicts the colliding key";

   disk_cache_destroy(cache);
}

/* To make sure we are not just using the inmemory cache index for the single
 * file cache we test adding and retriving cache items between two different
 * cache instances.
 */
static void
test_put_and_get_between_instances(const char *driver_id)
{
   char blob[] = "This is a blob of thirty-seven bytes";
   uint8_t blob_key[BLAKE3_KEY_LEN];
   char string[] = "While this string has thirty-four";
   uint8_t string_key[BLAKE3_KEY_LEN];
   char *result;
   size_t size;

#ifdef SHADER_CACHE_DISABLE_BY_DEFAULT
   os_set_option("MESA_SHADER_CACHE_DISABLE", "false", true);
#endif /* SHADER_CACHE_DISABLE_BY_DEFAULT */

   struct disk_cache *cache1 = disk_cache_create("test_between_instances",
                                                 driver_id, 0);
   struct disk_cache *cache2 = disk_cache_create("test_between_instances",
                                                 driver_id, 0);

   disk_cache_compute_key(cache1, blob, sizeof(blob), blob_key);

   /* Ensure that disk_cache_get returns nothing before anything is added. */
   result = (char *) disk_cache_get(cache1, blob_key, &size);
   EXPECT_EQ(result, nullptr) << "disk_cache_get(cache1) with non-existent item (pointer)";
   EXPECT_EQ(size, 0) << "disk_cache_get(cach1) with non-existent item (size)";

   result = (char *) disk_cache_get(cache2, blob_key, &size);
   EXPECT_EQ(result, nullptr) << "disk_cache_get(cache2) with non-existent item (pointer)";
   EXPECT_EQ(size, 0) << "disk_cache_get(cache2) with non-existent item (size)";

   /* Simple test of put and get. */
   disk_cache_put(cache1, blob_key, blob, sizeof(blob), NULL);

   /* disk_cache_put() hands things off to a thread so wait for it. */
   disk_cache_wait_for_idle(cache1);

   result = (char *) disk_cache_get(cache2, blob_key, &size);
   EXPECT_STREQ(blob, result) << "disk_cache_get(cache2) of existing item (pointer)";
   EXPECT_EQ(size, sizeof(blob)) << "disk_cache_get of(cache2) existing item (size)";

   free(result);

   /* Test put and get of a second item, via the opposite instances */
   disk_cache_compute_key(cache2, string, sizeof(string), string_key);
   disk_cache_put(cache2, string_key, string, sizeof(string), NULL);

   /* disk_cache_put() hands things off to a thread so wait for it. */
   disk_cache_wait_for_idle(cache2);

   result = (char *) disk_cache_get(cache1, string_key, &size);
   EXPECT_STREQ(result, string) << "2nd disk_cache_get(cache1) of existing item (pointer)";
   EXPECT_EQ(size, sizeof(string)) << "2nd disk_cache_get(cache1) of existing item (size)";

   free(result);

   disk_cache_destroy(cache1);
   disk_cache_destroy(cache2);
}

static void
test_put_and_get_between_instances_with_eviction(const char *driver_id)
{
   cache_key small_key[8], small_key2, big_key[2];
   struct disk_cache *cache[2];
   unsigned int i, n, k;
   uint8_t *small;
   uint8_t *big;
   char *result;
   size_t size;

#ifdef SHADER_CACHE_DISABLE_BY_DEFAULT
   os_set_option("MESA_SHADER_CACHE_DISABLE", "false", true);
#endif /* SHADER_CACHE_DISABLE_BY_DEFAULT */

   os_set_option("MESA_SHADER_CACHE_MAX_SIZE", "2K", true);

   cache[0] = disk_cache_create("test_between_instances_with_eviction", driver_id, 0);
   cache[1] = disk_cache_create("test_between_instances_with_eviction", driver_id, 0);

   uint8_t two_KB[2048] = { 0 };
   cache_key two_KB_key = { 'T', 'W', 'O', 'K', 'B' };

   /* Flush the database by adding the dummy 2KB entry */
   disk_cache_put(cache[0], two_KB_key, two_KB, sizeof(two_KB), NULL);
   disk_cache_wait_for_idle(cache[0]);

   int size_big = 1000;
   size_big -= sizeof(struct cache_entry_file_data);
   size_big -= mesa_cache_db_file_entry_size();
   size_big -= cache[0]->driver_keys_blob_size;
   size_big -= 4 + 8; /* cache_item_metadata size + room for alignment */

   for (i = 0; i < ARRAY_SIZE(big_key); i++) {
      big = (uint8_t *) malloc(size_big);
      memset(big, i, size_big);

      disk_cache_compute_key(cache[0], big, size_big, big_key[i]);
      disk_cache_put(cache[0], big_key[i], big, size_big, NULL);
      disk_cache_wait_for_idle(cache[0]);

      result = (char *) disk_cache_get(cache[0], big_key[i], &size);
      EXPECT_NE(result, nullptr) << "disk_cache_get with existent item (pointer)";
      EXPECT_EQ(size, size_big) << "disk_cache_get with existent item (size)";
      free(result);

      free(big);
   }

   int size_small = 256;
   size_small -= sizeof(struct cache_entry_file_data);
   size_small -= mesa_cache_db_file_entry_size();
   size_small -= cache[1]->driver_keys_blob_size;
   size_small -= 4 + 8; /* cache_item_metadata size + room for alignment */

   for (i = 0; i < ARRAY_SIZE(small_key); i++) {
      small = (uint8_t *) malloc(size_small);
      memset(small, i, size_small);

      disk_cache_compute_key(cache[1], small, size_small, small_key[i]);
      disk_cache_put(cache[1], small_key[i], small, size_small, NULL);
      disk_cache_wait_for_idle(cache[1]);

      /*
       * At first we added two 1000KB entries to cache[0]. Now, when first
       * 256KB entry is added, the two 1000KB entries are evicted because
       * at minimum cache_max_size/2 is evicted on overflow.
       *
       * All four 256KB entries stay in the cache.
       */
      for (k = 0; k < ARRAY_SIZE(cache); k++) {
         for (n = 0; n <= i; n++) {
            result = (char *) disk_cache_get(cache[k], big_key[0], &size);
            EXPECT_EQ(result, nullptr) << "disk_cache_get with non-existent item (pointer)";
            EXPECT_EQ(size, 0) << "disk_cache_get with non-existent item (size)";
            free(result);

            result = (char *) disk_cache_get(cache[k], big_key[1], &size);
            EXPECT_EQ(result, nullptr) << "disk_cache_get with non-existent item (pointer)";
            EXPECT_EQ(size, 0) << "disk_cache_get with non-existent item (size)";
            free(result);

            result = (char *) disk_cache_get(cache[k], small_key[n], &size);
            EXPECT_NE(result, nullptr) << "disk_cache_get of existing item (pointer)";
            EXPECT_EQ(size, size_small) << "disk_cache_get of existing item (size)";
            free(result);

            result = (char *) disk_cache_get(cache[k], small_key[n], &size);
            EXPECT_NE(result, nullptr) << "disk_cache_get of existing item (pointer)";
            EXPECT_EQ(size, size_small) << "disk_cache_get of existing item (size)";
            free(result);
         }
      }

      free(small);
   }

   small = (uint8_t *) malloc(size_small);
   memset(small, i, size_small);

   /* Add another 256KB entry. This will evict first five 256KB entries
    * of eight that we added previously. */
   disk_cache_compute_key(cache[0], small, size_small, small_key2);
   disk_cache_put(cache[0], small_key2, small, size_small, NULL);
   disk_cache_wait_for_idle(cache[0]);

   free(small);

   for (k = 0; k < ARRAY_SIZE(cache); k++) {
      result = (char *) disk_cache_get(cache[k], small_key2, &size);
      EXPECT_NE(result, nullptr) << "disk_cache_get of existing item (pointer)";
      EXPECT_EQ(size, size_small) << "disk_cache_get of existing item (size)";
      free(result);
   }

   for (i = 0, k = 0; k < ARRAY_SIZE(cache); k++) {
      for (n = 0; n < ARRAY_SIZE(small_key); n++) {
         result = (char *) disk_cache_get(cache[k], small_key[n], &size);
         if (!result)
            i++;
         free(result);
      }
   }

   EXPECT_EQ(i, 10) << "2x disk_cache_get with 5 non-existent 256KB items";

   disk_cache_destroy(cache[0]);
   disk_cache_destroy(cache[1]);
}

static void
test_put_big_sized_entry_to_empty_cache(const char *driver_id)
{
   static uint8_t blob[4096];
   uint8_t blob_key[BLAKE3_KEY_LEN];
   struct disk_cache *cache;
   char *result;
   size_t size;

#ifdef SHADER_CACHE_DISABLE_BY_DEFAULT
   os_set_option("MESA_SHADER_CACHE_DISABLE", "false", true);
#endif /* SHADER_CACHE_DISABLE_BY_DEFAULT */

   os_set_option("MESA_SHADER_CACHE_MAX_SIZE", "1K", true);
   cache = disk_cache_create("test", driver_id, 0);

   disk_cache_compute_key(cache, blob, sizeof(blob), blob_key);

   disk_cache_put(cache, blob_key, blob, sizeof(blob), NULL);
   disk_cache_wait_for_idle(cache);

   result = (char *) disk_cache_get(cache, blob_key, &size);
   EXPECT_NE(result, nullptr) << "disk_cache_get(cache) with existing item (pointer)";
   EXPECT_EQ(size, sizeof(blob)) << "disk_cache_get of(cache) existing item (size)";

   free(result);

   disk_cache_destroy(cache);
}
#endif /* ENABLE_SHADER_CACHE */

class Cache : public ::testing::Test {
protected:
   void *mem_ctx;

   Cache() {
      mem_ctx = ralloc_context(NULL);
   }
   ~Cache() {
      ralloc_free(mem_ctx);
   }
};

TEST_F(Cache, MultiFile)
{
   const char *driver_id;

#ifndef ENABLE_SHADER_CACHE
   GTEST_SKIP() << "ENABLE_SHADER_CACHE not defined.";
#else
   bool compress = true;

run_tests:
   os_set_option("MESA_DISK_CACHE_MULTI_FILE", "true", true);

   if (!compress)
      driver_id = "make_check_uncompressed";
   else
      driver_id = "make_check";

   test_disk_cache_create(mem_ctx, CACHE_DIR_NAME, driver_id);

   test_put_and_get(true, driver_id);

   test_put_key_and_get_key(driver_id);

   os_set_option("MESA_DISK_CACHE_MULTI_FILE", "false", true);

   int err = rmrf_local(CACHE_TEST_TMP);
   EXPECT_EQ(err, 0) << "Removing " CACHE_TEST_TMP " again";

   if (compress) {
      compress = false;
      goto run_tests;
   }
#endif
}

TEST_F(Cache, SingleFile)
{
   const char *driver_id;

#ifndef ENABLE_SHADER_CACHE
   GTEST_SKIP() << "ENABLE_SHADER_CACHE not defined.";
#else
   bool compress = true;

run_tests:
   os_set_option("MESA_DISK_CACHE_SINGLE_FILE", "true", true);

   if (!compress)
      driver_id = "make_check_uncompressed";
   else
      driver_id = "make_check";

   test_disk_cache_create(mem_ctx, CACHE_DIR_NAME_SF, driver_id);

   /* We skip testing cache size limit as the single file cache currently
    * doesn't have any functionality to enforce cache size limits.
    */
   test_put_and_get(false, driver_id);

   test_put_key_and_get_key(driver_id);

   test_put_and_get_between_instances(driver_id);

   os_set_option("MESA_DISK_CACHE_SINGLE_FILE", "false", true);

   int err = rmrf_local(CACHE_TEST_TMP);
   EXPECT_EQ(err, 0) << "Removing " CACHE_TEST_TMP " again";

   if (compress) {
      compress = false;
      goto run_tests;
   }
#endif
}

TEST_F(Cache, Database)
{
   const char *driver_id = "make_check_uncompressed";

#ifndef ENABLE_SHADER_CACHE
   GTEST_SKIP() << "ENABLE_SHADER_CACHE not defined.";
#else
   os_set_option("MESA_DISK_CACHE_MULTI_FILE", "false", true);
   os_set_option("MESA_DISK_CACHE_DATABASE_NUM_PARTS", "1", true);
   os_set_option("MESA_DISK_CACHE_DATABASE", "true", true);

   test_disk_cache_create(mem_ctx, CACHE_DIR_NAME_DB, driver_id);

   /* We skip testing cache size limit as the single file cache compresses
    * data much better than the multi-file cache, which results in the
    * failing tests of the cache eviction function. We we will test the
    * eviction separately with the disabled compression.
    */
   test_put_and_get(false, driver_id);

   int err = rmrf_local(CACHE_TEST_TMP);
   EXPECT_EQ(err, 0) << "Removing " CACHE_TEST_TMP " again";

   test_disk_cache_create(mem_ctx, CACHE_DIR_NAME_DB, driver_id);

   test_put_and_get(true, driver_id);

   test_put_key_and_get_key(driver_id);

   test_put_and_get_between_instances(driver_id);

   test_put_and_get_between_instances_with_eviction(driver_id);

   test_put_big_sized_entry_to_empty_cache(driver_id);

   os_set_option("MESA_DISK_CACHE_DATABASE", "false", true);
   os_unset_option("MESA_DISK_CACHE_DATABASE_NUM_PARTS");

   err = rmrf_local(CACHE_TEST_TMP);
   EXPECT_EQ(err, 0) << "Removing " CACHE_TEST_TMP " again";
#endif
}

TEST_F(Cache, Combined)
{
   const char *driver_id = "make_check";
   char blob[] = "This is a RO blob";
   char blob2[] = "This is a RW blob";
   uint8_t dummy_key[BLAKE3_KEY_LEN] = { 0 };
   uint8_t blob_key[BLAKE3_KEY_LEN];
   uint8_t blob_key2[BLAKE3_KEY_LEN];
   char foz_rw_idx_file[1024];
   char foz_ro_idx_file[1024];
   char foz_rw_file[1024];
   char foz_ro_file[1024];
   char *result;
   size_t size;

#ifndef ENABLE_SHADER_CACHE
   GTEST_SKIP() << "ENABLE_SHADER_CACHE not defined.";
#else
   os_set_option("MESA_DISK_CACHE_SINGLE_FILE", "true", true);
   os_set_option("MESA_DISK_CACHE_MULTI_FILE", "true", true);
   os_set_option("MESA_DISK_CACHE_DATABASE", "false", true);

#ifdef SHADER_CACHE_DISABLE_BY_DEFAULT
   os_set_option("MESA_SHADER_CACHE_DISABLE", "false", true);
#endif /* SHADER_CACHE_DISABLE_BY_DEFAULT */

   /* Enable Fossilize read-write cache. */
   os_set_option("MESA_DISK_CACHE_COMBINE_RW_WITH_RO_FOZ", "true", true);

   test_disk_cache_create(mem_ctx, CACHE_DIR_NAME_SF, driver_id);

   /* Create Fossilize writable cache. */
   struct disk_cache *cache_sf_wr = disk_cache_create("combined_test",
                                                      driver_id, 0);

   disk_cache_compute_key(cache_sf_wr, blob, sizeof(blob), blob_key);
   disk_cache_compute_key(cache_sf_wr, blob2, sizeof(blob2), blob_key2);

   /* Ensure that disk_cache_get returns nothing before anything is added. */
   result = (char *) disk_cache_get(cache_sf_wr, blob_key, &size);
   EXPECT_EQ(result, nullptr) << "disk_cache_get with non-existent item (pointer)";
   EXPECT_EQ(size, 0) << "disk_cache_get with non-existent item (size)";

   /* Put blob entry to the cache. */
   disk_cache_put(cache_sf_wr, blob_key, blob, sizeof(blob), NULL);
   disk_cache_wait_for_idle(cache_sf_wr);

   result = (char *) disk_cache_get(cache_sf_wr, blob_key, &size);
   EXPECT_STREQ(blob, result) << "disk_cache_get of existing item (pointer)";
   EXPECT_EQ(size, sizeof(blob)) << "disk_cache_get of existing item (size)";
   free(result);

   /* Rename file foz_cache.foz -> ro_cache.foz */
   snprintf(foz_rw_file, sizeof(foz_rw_file), "%s/foz_cache.foz", cache_sf_wr->path);
   snprintf(foz_ro_file, sizeof(foz_ro_file), "%s/ro_cache.foz", cache_sf_wr->path);
   EXPECT_EQ(rename(foz_rw_file, foz_ro_file), 0) << "foz_cache.foz renaming failed";

   /* Rename file foz_cache_idx.foz -> ro_cache_idx.foz */
   snprintf(foz_rw_idx_file, sizeof(foz_rw_idx_file), "%s/foz_cache_idx.foz", cache_sf_wr->path);
   snprintf(foz_ro_idx_file, sizeof(foz_ro_idx_file), "%s/ro_cache_idx.foz", cache_sf_wr->path);
   EXPECT_EQ(rename(foz_rw_idx_file, foz_ro_idx_file), 0) << "foz_cache_idx.foz renaming failed";

   disk_cache_destroy(cache_sf_wr);

   /* Disable Fossilize read-write cache. */
   os_set_option("MESA_DISK_CACHE_COMBINE_RW_WITH_RO_FOZ", "false", true);

   /* Set up Fossilize read-only cache. */
   os_set_option("MESA_DISK_CACHE_COMBINE_RW_WITH_RO_FOZ", "true", true);
   os_set_option("MESA_DISK_CACHE_READ_ONLY_FOZ_DBS", "ro_cache", true);

   /* Create FOZ cache that fetches the RO cache. Note that this produces
    * empty RW cache files. */
   struct disk_cache *cache_sf_ro = disk_cache_create("combined_test",
                                                      driver_id, 0);

   /* Blob entry must present because it shall be retrieved from the
    * ro_cache.foz */
   result = (char *) disk_cache_get(cache_sf_ro, blob_key, &size);
   EXPECT_STREQ(blob, result) << "disk_cache_get of existing item (pointer)";
   EXPECT_EQ(size, sizeof(blob)) << "disk_cache_get of existing item (size)";
   free(result);

   disk_cache_destroy(cache_sf_ro);

   /* Remove empty FOZ RW cache files created above. We only need RO cache. */
   EXPECT_EQ(unlink(foz_rw_file), 0);
   EXPECT_EQ(unlink(foz_rw_idx_file), 0);

   os_set_option("MESA_DISK_CACHE_SINGLE_FILE", "false", true);
   os_set_option("MESA_DISK_CACHE_MULTI_FILE", "false", true);
   os_set_option("MESA_DISK_CACHE_DATABASE", "true", true);

   /* Create MESA-DB cache with enabled retrieval from the read-only
    * cache. */
   struct disk_cache *cache_mesa_db = disk_cache_create("combined_test",
                                                        driver_id, 0);

   /* Dummy entry must not present in any of the caches. Foz cache
    * reloads index if cache entry is missing.  This is a sanity-check
    * for foz_read_entry(), it should work properly with a disabled
    * FOZ RW cache. */
   result = (char *) disk_cache_get(cache_mesa_db, dummy_key, &size);
   EXPECT_EQ(result, nullptr) << "disk_cache_get with non-existent item (pointer)";
   EXPECT_EQ(size, 0) << "disk_cache_get with non-existent item (size)";

   /* Blob entry must present because it shall be retrieved from the
    * read-only cache. */
   result = (char *) disk_cache_get(cache_mesa_db, blob_key, &size);
   EXPECT_STREQ(blob, result) << "disk_cache_get of existing item (pointer)";
   EXPECT_EQ(size, sizeof(blob)) << "disk_cache_get of existing item (size)";
   free(result);

   /* Blob2 entry must not present in any of the caches. */
   result = (char *) disk_cache_get(cache_mesa_db, blob_key2, &size);
   EXPECT_EQ(result, nullptr) << "disk_cache_get with non-existent item (pointer)";
   EXPECT_EQ(size, 0) << "disk_cache_get with non-existent item (size)";

   /* Put blob2 entry to the cache. */
   disk_cache_put(cache_mesa_db, blob_key2, blob2, sizeof(blob2), NULL);
   disk_cache_wait_for_idle(cache_mesa_db);

   /* Blob2 entry must present because it shall be retrieved from the
    * read-write cache. */
   result = (char *) disk_cache_get(cache_mesa_db, blob_key2, &size);
   EXPECT_STREQ(blob2, result) << "disk_cache_get of existing item (pointer)";
   EXPECT_EQ(size, sizeof(blob2)) << "disk_cache_get of existing item (size)";
   free(result);

   disk_cache_destroy(cache_mesa_db);

   /* Disable read-only cache. */
   os_set_option("MESA_DISK_CACHE_COMBINE_RW_WITH_RO_FOZ", "false", true);

   /* Create MESA-DB cache with disabled retrieval from the
    * read-only cache. */
   cache_mesa_db = disk_cache_create("combined_test", driver_id, 0);

   /* Blob2 entry must present because it shall be retrieved from the
    * MESA-DB cache. */
   result = (char *) disk_cache_get(cache_mesa_db, blob_key2, &size);
   EXPECT_STREQ(blob2, result) << "disk_cache_get of existing item (pointer)";
   EXPECT_EQ(size, sizeof(blob2)) << "disk_cache_get of existing item (size)";
   free(result);

   disk_cache_destroy(cache_mesa_db);

   /* Create MESA-DB cache with disabled retrieval from the read-only
    * cache. */
   cache_mesa_db = disk_cache_create("combined_test", driver_id, 0);

   /* Blob entry must not present in the cache because we disable the
    * read-only cache. */
   result = (char *) disk_cache_get(cache_mesa_db, blob_key, &size);
   EXPECT_EQ(result, nullptr) << "disk_cache_get with non-existent item (pointer)";
   EXPECT_EQ(size, 0) << "disk_cache_get with non-existent item (size)";

   disk_cache_destroy(cache_mesa_db);

   /* Create default multi-file cache. */
   os_set_option("MESA_DISK_CACHE_DATABASE", "false", true);
   os_set_option("MESA_DISK_CACHE_MULTI_FILE", "true", true);

   /* Enable read-only cache. */
   os_set_option("MESA_DISK_CACHE_COMBINE_RW_WITH_RO_FOZ", "true", true);

   /* Create multi-file cache with enabled retrieval from the
    * read-only cache. */
   struct disk_cache *cache_multifile = disk_cache_create("combined_test",
                                                          driver_id, 0);

   /* Blob entry must present because it shall be retrieved from the
    * read-only cache. */
   result = (char *) disk_cache_get(cache_multifile, blob_key, &size);
   EXPECT_STREQ(blob, result) << "disk_cache_get of existing item (pointer)";
   EXPECT_EQ(size, sizeof(blob)) << "disk_cache_get of existing item (size)";
   free(result);

   /* Blob2 entry must not present in any of the caches. */
   result = (char *) disk_cache_get(cache_multifile, blob_key2, &size);
   EXPECT_EQ(result, nullptr) << "disk_cache_get with non-existent item (pointer)";
   EXPECT_EQ(size, 0) << "disk_cache_get with non-existent item (size)";

   /* Put blob2 entry to the cache. */
   disk_cache_put(cache_multifile, blob_key2, blob2, sizeof(blob2), NULL);
   disk_cache_wait_for_idle(cache_multifile);

   /* Blob2 entry must present because it shall be retrieved from the
    * read-write cache. */
   result = (char *) disk_cache_get(cache_multifile, blob_key2, &size);
   EXPECT_STREQ(blob2, result) << "disk_cache_get of existing item (pointer)";
   EXPECT_EQ(size, sizeof(blob2)) << "disk_cache_get of existing item (size)";
   free(result);

   disk_cache_destroy(cache_multifile);

   /* Disable read-only cache. */
   os_set_option("MESA_DISK_CACHE_COMBINE_RW_WITH_RO_FOZ", "false", true);
   os_unset_option("MESA_DISK_CACHE_READ_ONLY_FOZ_DBS");

   /* Create multi-file cache with disabled retrieval from the
    * read-only cache. */
   cache_multifile = disk_cache_create("combined_test", driver_id, 0);

   /* Blob entry must not present in the cache because we disabled the
    * read-only cache. */
   result = (char *) disk_cache_get(cache_multifile, blob_key, &size);
   EXPECT_EQ(result, nullptr) << "disk_cache_get with non-existent item (pointer)";
   EXPECT_EQ(size, 0) << "disk_cache_get with non-existent item (size)";

   /* Blob2 entry must present because it shall be retrieved from the
    * read-write cache. */
   result = (char *) disk_cache_get(cache_multifile, blob_key2, &size);
   EXPECT_STREQ(blob2, result) << "disk_cache_get of existing item (pointer)";
   EXPECT_EQ(size, sizeof(blob2)) << "disk_cache_get of existing item (size)";
   free(result);

   disk_cache_destroy(cache_multifile);

   os_unset_option("MESA_DISK_CACHE_SINGLE_FILE");
   os_unset_option("MESA_DISK_CACHE_MULTI_FILE");
   os_unset_option("MESA_DISK_CACHE_DATABASE");

   int err = rmrf_local(CACHE_TEST_TMP);
   EXPECT_EQ(err, 0) << "Removing " CACHE_TEST_TMP " again";
#endif
}

TEST_F(Cache, List)
{
#ifndef ENABLE_SHADER_CACHE
   GTEST_SKIP() << "ENABLE_SHADER_CACHE not defined.";
#else
#ifndef FOZ_DB_UTIL_DYNAMIC_LIST
   GTEST_SKIP() << "FOZ_DB_UTIL_DYNAMIC_LIST not supported";
#else
   const char *driver_id = "make_check";
   char blob[] = "This is a RO blob";
   uint8_t blob_key[BLAKE3_KEY_LEN];
   char foz_rw_idx_file[1024];
   char foz_ro_idx_file[1024];
   char foz_rw_file[1024];
   char foz_ro_file[1024];
   char *result;
   size_t size;

   os_set_option("MESA_DISK_CACHE_SINGLE_FILE", "true", true);

#ifdef SHADER_CACHE_DISABLE_BY_DEFAULT
   os_set_option("MESA_SHADER_CACHE_DISABLE", "false", true);
#endif /* SHADER_CACHE_DISABLE_BY_DEFAULT */

   test_disk_cache_create(mem_ctx, CACHE_DIR_NAME_SF, driver_id);

   /* Create ro files for testing */
   /* Create Fossilize writable cache. */
   struct disk_cache *cache_sf_wr =
      disk_cache_create("list_test", driver_id, 0);

   disk_cache_compute_key(cache_sf_wr, blob, sizeof(blob), blob_key);

   /* Ensure that disk_cache_get returns nothing before anything is added. */
   result = (char *)disk_cache_get(cache_sf_wr, blob_key, &size);
   EXPECT_EQ(result, nullptr)
      << "disk_cache_get with non-existent item (pointer)";
   EXPECT_EQ(size, 0) << "disk_cache_get with non-existent item (size)";

   /* Put blob entry to the cache. */
   disk_cache_put(cache_sf_wr, blob_key, blob, sizeof(blob), NULL);
   disk_cache_wait_for_idle(cache_sf_wr);

   result = (char *)disk_cache_get(cache_sf_wr, blob_key, &size);
   EXPECT_STREQ(blob, result) << "disk_cache_get of existing item (pointer)";
   EXPECT_EQ(size, sizeof(blob)) << "disk_cache_get of existing item (size)";
   free(result);

   /* Rename file foz_cache.foz -> ro_cache.foz */
   sprintf(foz_rw_file, "%s/foz_cache.foz", cache_sf_wr->path);
   sprintf(foz_ro_file, "%s/ro_cache.foz", cache_sf_wr->path);
   EXPECT_EQ(rename(foz_rw_file, foz_ro_file), 0)
      << "foz_cache.foz renaming failed";

   /* Rename file foz_cache_idx.foz -> ro_cache_idx.foz */
   sprintf(foz_rw_idx_file, "%s/foz_cache_idx.foz", cache_sf_wr->path);
   sprintf(foz_ro_idx_file, "%s/ro_cache_idx.foz", cache_sf_wr->path);
   EXPECT_EQ(rename(foz_rw_idx_file, foz_ro_idx_file), 0)
      << "foz_cache_idx.foz renaming failed";

   disk_cache_destroy(cache_sf_wr);

   const char *list_filename = CACHE_TEST_TMP "/foz_dbs_list.txt";
   os_set_option("MESA_DISK_CACHE_READ_ONLY_FOZ_DBS_DYNAMIC_LIST", list_filename, true);

   /* Create new empty file */
   FILE *list_file = fopen(list_filename, "w");
   fputs("ro_cache\n", list_file);
   fclose(list_file);

   /* Create Fossilize writable cache. */
   struct disk_cache *cache_sf = disk_cache_create("list_test", driver_id, 0);

   /* Blob entry must present because it shall be retrieved from the
    * ro_cache.foz loaded from list at creation time */
   result = (char *)disk_cache_get(cache_sf, blob_key, &size);
   EXPECT_STREQ(blob, result) << "disk_cache_get of existing item (pointer)";
   EXPECT_EQ(size, sizeof(blob)) << "disk_cache_get of existing item (size)";
   free(result);

   disk_cache_destroy(cache_sf);
   remove(list_filename);

   /* Test loading from a list populated at runtime */
   /* Create new empty file */
   list_file = fopen(list_filename, "w");
   fclose(list_file);

   /* Create Fossilize writable cache. */
   cache_sf = disk_cache_create("list_test", driver_id, 0);

   /* Ensure that disk_cache returns nothing before list file is populated */
   result = (char *)disk_cache_get(cache_sf, blob_key, &size);
   EXPECT_EQ(result, nullptr)
      << "disk_cache_get with non-existent item (pointer)";
   EXPECT_EQ(size, 0) << "disk_cache_get with non-existent item (size)";

   /* Add ro_cache to list file for loading */
   list_file = fopen(list_filename, "a");
   fputs("ro_cache\n", list_file);

   /* Close the list file for writing and wait for the updater to finish
    * updating from the list.
    */
   close_and_wait_for_list(list_file, list_filename);
   result = (char *)disk_cache_get(cache_sf, blob_key, &size);

   /* Blob entry must present because it shall be retrieved from the
    * ro_cache.foz loaded from list at runtime */
   EXPECT_STREQ(blob, result) << "disk_cache_get of existing item (pointer)";
   EXPECT_EQ(size, sizeof(blob)) << "disk_cache_get of existing item (size)";
   free(result);

   disk_cache_destroy(cache_sf);
   remove(list_filename);

   /* Test loading from a list with some invalid files */
   /* Create new empty file */
   list_file = fopen(list_filename, "w");
   fclose(list_file);

   /* Create Fossilize writable cache. */
   cache_sf = disk_cache_create("list_test", driver_id, 0);

   /* Ensure that disk_cache returns nothing before list file is populated */
   result = (char *)disk_cache_get(cache_sf, blob_key, &size);
   EXPECT_EQ(result, nullptr)
      << "disk_cache_get with non-existent item (pointer)";
   EXPECT_EQ(size, 0) << "disk_cache_get with non-existent item (size)";

   /* Add non-existant list files for loading */
   list_file = fopen(list_filename, "a");
   fputs("no_cache\n", list_file);
   fputs("no_cache2\n", list_file);
   fputs("no_cache/no_cache3\n", list_file);
   /* Add ro_cache to list file for loading */
   fputs("ro_cache\n", list_file);

   /* Close the list file for writing and wait for the updater to finish
    * updating from the list.
    */
   close_and_wait_for_list(list_file, list_filename);
   result = (char *)disk_cache_get(cache_sf, blob_key, &size);

   /* Blob entry must present because it shall be retrieved from the
    * ro_cache.foz loaded from list at runtime despite invalid files
    * in the list */
   EXPECT_STREQ(blob, result) << "disk_cache_get of existing item (pointer)";
   EXPECT_EQ(size, sizeof(blob)) << "disk_cache_get of existing item (size)";
   free(result);

   disk_cache_destroy(cache_sf);
   remove(list_filename);

   int err = rmrf_local(CACHE_TEST_TMP);
   EXPECT_EQ(err, 0) << "Removing " CACHE_TEST_TMP " again";

   os_unset_option("MESA_DISK_CACHE_SINGLE_FILE");
   os_unset_option("MESA_DISK_CACHE_READ_ONLY_FOZ_DBS_DYNAMIC_LIST");
#endif /* FOZ_DB_UTIL_DYNAMIC_LIST */
#endif /* ENABLE_SHADER_CACHE */
}

static void
test_multipart_eviction(const char *driver_id)
{
   const unsigned int entry_size = 512;
   uint8_t blobs[7][entry_size];
   cache_key keys[7];
   unsigned int i;
   char *result;
   size_t size;

   os_set_option("MESA_SHADER_CACHE_MAX_SIZE", "3K", true);
   os_set_option("MESA_DISK_CACHE_DATABASE_EVICTION_SCORE_2X_PERIOD", "1", true);

   struct disk_cache *cache = disk_cache_create("test", driver_id, 0);

   unsigned int entry_file_size = entry_size;
   entry_file_size -= sizeof(struct cache_entry_file_data);
   entry_file_size -= mesa_cache_db_file_entry_size();
   entry_file_size -= cache->driver_keys_blob_size;
   entry_file_size -= 4 + 8; /* cache_item_metadata size + room for alignment */

   /*
    * 1. Allocate 3KB cache in 3 parts, each part is 1KB
    * 2. Fill up cache with six 512K entries
    * 3. Touch entries of the first part, which will bump last_access_time
    *    of the first two cache entries
    * 4. Insert seventh 512K entry that will cause eviction of the second part
    * 5. Check that second entry of the second part gone due to eviction and
    *    others present
    */

   /* Fill up cache with six 512K entries. */
   for (i = 0; i < 6; i++) {
      memset(blobs[i], i, entry_file_size);

      disk_cache_compute_key(cache,  blobs[i], entry_file_size, keys[i]);
      disk_cache_put(cache, keys[i], blobs[i], entry_file_size, NULL);
      disk_cache_wait_for_idle(cache);

      result = (char *) disk_cache_get(cache, keys[i], &size);
      EXPECT_NE(result, nullptr) << "disk_cache_get with existent item (pointer)";
      EXPECT_EQ(size, entry_file_size) << "disk_cache_get with existent item (size)";
      free(result);

      /* Ensure that cache entries will have distinct last_access_time
       * during testing.
       */
      if (i % 2 == 0)
         usleep(100000);
   }

   /* Touch entries of the first part. Second part becomes outdated */
   for (i = 0; i < 2; i++) {
      result = (char *) disk_cache_get(cache, keys[i], &size);
      EXPECT_NE(result, nullptr) << "disk_cache_get with existent item (pointer)";
      EXPECT_EQ(size, entry_file_size) << "disk_cache_get with existent item (size)";
      free(result);
   }

   /* Insert seventh entry. */
   memset(blobs[6], 6, entry_file_size);
   disk_cache_compute_key(cache,  blobs[6], entry_file_size, keys[6]);
   disk_cache_put(cache, keys[6], blobs[6], entry_file_size, NULL);
   disk_cache_wait_for_idle(cache);

   /* Check whether third entry of the second part gone and others present. */
   for (i = 0; i < ARRAY_SIZE(blobs); i++) {
      result = (char *) disk_cache_get(cache, keys[i], &size);
      if (i == 2) {
         EXPECT_EQ(result, nullptr) << "disk_cache_get with non-existent item (pointer)";
      } else {
         EXPECT_NE(result, nullptr) << "disk_cache_get with existent item (pointer)";
         EXPECT_EQ(size, entry_file_size) << "disk_cache_get with existent item (size)";
      }
      free(result);
   }

   disk_cache_destroy(cache);
}

TEST_F(Cache, DatabaseMultipartEviction)
{
   const char *driver_id = "make_check_uncompressed";

#ifndef ENABLE_SHADER_CACHE
   GTEST_SKIP() << "ENABLE_SHADER_CACHE not defined.";
#else
   os_set_option("MESA_DISK_CACHE_MULTI_FILE", "false", true);
   os_set_option("MESA_DISK_CACHE_DATABASE_NUM_PARTS", "3", true);
   os_set_option("MESA_DISK_CACHE_DATABASE", "true", true);

   test_disk_cache_create(mem_ctx, CACHE_DIR_NAME_DB, driver_id);

   test_multipart_eviction(driver_id);

   os_unset_option("MESA_DISK_CACHE_DATABASE_NUM_PARTS");
   os_unset_option("MESA_DISK_CACHE_DATABASE");

   int err = rmrf_local(CACHE_TEST_TMP);
   EXPECT_EQ(err, 0) << "Removing " CACHE_TEST_TMP " again";
#endif
}

static void
test_put_and_get_disabled(const char *driver_id)
{
   struct disk_cache *cache;
   char blob[] = "This is a blob of thirty-seven bytes";
   uint8_t blob_key[BLAKE3_KEY_LEN];
   char *result;
   size_t size;

   cache = disk_cache_create("test", driver_id, 0);

   disk_cache_compute_key(cache, blob, sizeof(blob), blob_key);

   /* Ensure that disk_cache_get returns nothing before anything is added. */
   result = (char *) disk_cache_get(cache, blob_key, &size);
   EXPECT_EQ(result, nullptr) << "disk_cache_get with non-existent item (pointer)";
   EXPECT_EQ(size, 0) << "disk_cache_get with non-existent item (size)";

   /* Simple test of put and get. */
   disk_cache_put(cache, blob_key, blob, sizeof(blob), NULL);

   /* disk_cache_put() hands things off to a thread so wait for it. */
   disk_cache_wait_for_idle(cache);

   result = (char *) disk_cache_get(cache, blob_key, &size);
   EXPECT_STREQ(result, nullptr) << "disk_cache_get of existing item (pointer)";
   EXPECT_EQ(size, 0) << "disk_cache_get of existing item (size)";

   disk_cache_destroy(cache);
}

TEST_F(Cache, Disabled)
{
   const char *driver_id = "make_check";

#ifndef ENABLE_SHADER_CACHE
   GTEST_SKIP() << "ENABLE_SHADER_CACHE not defined.";
#else
   os_set_option("MESA_DISK_CACHE_SINGLE_FILE", "true", true);

#ifdef SHADER_CACHE_DISABLE_BY_DEFAULT
   os_set_option("MESA_SHADER_CACHE_DISABLE", "false", true);
#endif /* SHADER_CACHE_DISABLE_BY_DEFAULT */

   test_disk_cache_create(mem_ctx, CACHE_DIR_NAME_SF, driver_id);

   test_put_and_get(false, driver_id);

   os_set_option("MESA_SHADER_CACHE_DISABLE", "true", true);

   test_put_and_get_disabled(driver_id);

   os_set_option("MESA_SHADER_CACHE_DISABLE", "false", true);
   os_set_option("MESA_DISK_CACHE_SINGLE_FILE", "false", true);

   int err = rmrf_local(CACHE_TEST_TMP);
   EXPECT_EQ(err, 0) << "Removing " CACHE_TEST_TMP " again";
#endif
}

TEST_F(Cache, DoNotDeleteNewCache)
{
#ifndef ENABLE_SHADER_CACHE
   GTEST_SKIP() << "ENABLE_SHADER_CACHE not defined.";
#else

#ifdef SHADER_CACHE_DISABLE_BY_DEFAULT
   os_set_option("MESA_SHADER_CACHE_DISABLE", "false", true);
#endif /* SHADER_CACHE_DISABLE_BY_DEFAULT */

   char dir_template[] = "/tmp/tmpdir.XXXXXX";
   char *dir_name = mkdtemp(dir_template);
   ASSERT_NE(dir_name, nullptr);

   char cache_dir_name[256];
   snprintf(cache_dir_name, sizeof(cache_dir_name), "%s/mesa_shader_cache", dir_name);
   mkdir(cache_dir_name, 0755);

   os_set_option("MESA_SHADER_CACHE_DIR", dir_name, true);

   disk_cache_delete_old_cache();

   struct stat st;
   EXPECT_EQ(stat(cache_dir_name, &st), 0);

   os_unset_option("MESA_SHADER_CACHE_DIR");
   rmdir(cache_dir_name);
   rmdir(dir_name);
#endif
}

TEST_F(Cache, DoNotDeleteCacheWithNewMarker)
{
#ifndef ENABLE_SHADER_CACHE
   GTEST_SKIP() << "ENABLE_SHADER_CACHE not defined.";
#else

#ifdef SHADER_CACHE_DISABLE_BY_DEFAULT
   os_set_option("MESA_SHADER_CACHE_DISABLE", "false", true);
#endif /* SHADER_CACHE_DISABLE_BY_DEFAULT */

   char dir_template[] = "/tmp/tmpdir.XXXXXX";
   char *dir_name = mkdtemp(dir_template);
   ASSERT_NE(dir_name, nullptr);

   char cache_dir_name[240];
   snprintf(cache_dir_name, sizeof(cache_dir_name), "%s/mesa_shader_cache", dir_name);
   mkdir(cache_dir_name, 0755);

   char file_name[256];
   snprintf(file_name, sizeof(file_name), "%s/marker", cache_dir_name);

   FILE *file = fopen(file_name, "w");
   fclose(file);

   os_set_option("MESA_SHADER_CACHE_DIR", dir_name, true);

   disk_cache_delete_old_cache();

   struct stat st;
   EXPECT_EQ(stat(cache_dir_name, &st), 0);

   os_unset_option("MESA_SHADER_CACHE_DIR");
   unlink(file_name);
   rmdir(cache_dir_name);
   rmdir(dir_name);
#endif
}

TEST_F(Cache, DeleteOldCache)
{
#ifndef ENABLE_SHADER_CACHE
   GTEST_SKIP() << "ENABLE_SHADER_CACHE not defined.";
#else

#ifdef SHADER_CACHE_DISABLE_BY_DEFAULT
   os_set_option("MESA_SHADER_CACHE_DISABLE", "false", true);
#endif /* SHADER_CACHE_DISABLE_BY_DEFAULT */

   char dir_template[] = "/tmp/tmpdir.XXXXXX";
   char *dir_name = mkdtemp(dir_template);
   ASSERT_NE(dir_name, nullptr) << "Creating temporary directory failed";

   char cache_dir_name[240];
   snprintf(cache_dir_name, sizeof(cache_dir_name), "%s/mesa_shader_cache", dir_name);
   mkdir(cache_dir_name, 0755);

   char file_name[256];
   snprintf(file_name, sizeof(file_name), "%s/marker", cache_dir_name);

   FILE *file = fopen(file_name, "w");
   fclose(file);

   struct utimbuf utime_buf = { };
   EXPECT_EQ(utime(file_name, &utime_buf), 0);


   os_set_option("MESA_SHADER_CACHE_DIR", dir_name, true);

   disk_cache_delete_old_cache();

   struct stat st;
   EXPECT_NE(stat(cache_dir_name, &st), 0);
   EXPECT_EQ(errno, ENOENT);

   os_unset_option("MESA_SHADER_CACHE_DIR");
   unlink(file_name);
   rmdir(cache_dir_name);
   rmdir(dir_name);
#endif
}

class CacheMultiFile : public Cache {
protected:
   struct disk_cache *cache = NULL;

   void SetUp() override {
#if !defined(ENABLE_SHADER_CACHE)
      GTEST_SKIP() << "ENABLE_SHADER_CACHE not defined.";
#elif DETECT_OS_ANDROID
      GTEST_SKIP() << "Android does not write cache files to disk.";
#else
#ifdef SHADER_CACHE_DISABLE_BY_DEFAULT
      os_set_option("MESA_SHADER_CACHE_DISABLE", "false", true);
#endif /* SHADER_CACHE_DISABLE_BY_DEFAULT */
      os_set_option("MESA_DISK_CACHE_MULTI_FILE", "true", true);
      os_set_option("MESA_SHADER_CACHE_DIR", CACHE_TEST_TMP "/multi-file", true);
      os_set_option("MESA_SHADER_CACHE_MAX_SIZE", "1M", true);
      open_cache();
#endif
   }

   void TearDown() override {
#if defined(ENABLE_SHADER_CACHE) && !DETECT_OS_ANDROID
      close_cache();
      os_set_option("MESA_DISK_CACHE_MULTI_FILE", "false", true);
      os_unset_option("MESA_SHADER_CACHE_DIR");
      os_unset_option("MESA_SHADER_CACHE_MAX_SIZE");
      rmrf_local(CACHE_TEST_TMP);
#endif
   }

#if defined(ENABLE_SHADER_CACHE) && !DETECT_OS_ANDROID
   void open_cache() {
      cache = disk_cache_create("test", "make_check", 0);
      ASSERT_NE(cache, nullptr);
      ASSERT_EQ(cache->type, DISK_CACHE_MULTI_FILE);
   }

   void close_cache() {
      if (cache)
         disk_cache_destroy(cache);
      cache = NULL;
   }

   uint64_t counter() {
      return p_atomic_read(&cache->size->value);
   }

   void set_counter(uint64_t value) {
      p_atomic_set(&cache->size->value, value);
   }

   void put(unsigned seed, size_t size, cache_key key) {
      uint8_t *data = (uint8_t *) malloc(size);
      uint32_t state = seed * 2654435761u + 1;
      for (size_t i = 0; i < size; i++) {
         state = state * 1664525u + 1013904223u;
         data[i] = state >> 24;
      }
      disk_cache_compute_key(cache, data, size, key);
      disk_cache_put(cache, key, data, size, NULL);
      disk_cache_wait_for_idle(cache);
      free(data);
   }

   std::string item_dir(const cache_key key) {
      char *filename = disk_cache_get_cache_filename(cache, key);
      std::string dir(filename);
      free(filename);
      return dir.substr(0, dir.rfind('/'));
   }

   uint64_t size_on_disk() {
      uint64_t total = 0;
      for (unsigned i = 0; i < 256; i++) {
         char *dir_path = ralloc_asprintf(mem_ctx, "%s/%02x", cache->path, i);
         DIR *dir = opendir(dir_path);
         if (!dir)
            continue;
         struct dirent *ent;
         while ((ent = readdir(dir)) != NULL) {
            size_t len = strlen(ent->d_name);
            if (len >= 4 && !strcmp(&ent->d_name[len - 4], ".tmp"))
               continue;
            char *path = ralloc_asprintf(mem_ctx, "%s/%s", dir_path, ent->d_name);
            struct stat sb;
            if (lstat(path, &sb) == 0 && S_ISREG(sb.st_mode))
               total += (uint64_t) sb.st_blocks * 512;
         }
         closedir(dir);
      }
      return total;
   }

   void write_counter_to_index(const std::string &cache_path, uint64_t value) {
      std::string index = cache_path + "/index";
      int fd = open(index.c_str(), O_WRONLY);
      ASSERT_NE(fd, -1);
      ASSERT_EQ(pwrite(fd, &value, sizeof(value), 0), (ssize_t) sizeof(value));
      close(fd);
   }

   void reopen_with_counter(uint64_t value) {
      std::string cache_path(cache->path);
      close_cache();
      write_counter_to_index(cache_path, value);
      open_cache();
   }

   static void create_file(const std::string &path, time_t mtime) {
      static const char data[4096] = { 1 };
      int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
      ASSERT_NE(fd, -1);
      ASSERT_EQ(write(fd, data, sizeof(data)), (ssize_t) sizeof(data));
      close(fd);
      struct utimbuf times = { mtime, mtime };
      ASSERT_EQ(utime(path.c_str(), &times), 0);
   }

   static bool is_readable_memory(const void *ptr) {
      int fds[2];
      if (pipe(fds) != 0)
         return false;
      ssize_t ret = write(fds[1], ptr, 1);
      close(fds[0]);
      close(fds[1]);
      return ret == 1;
   }
#endif
};

#if defined(ENABLE_SHADER_CACHE) && !DETECT_OS_ANDROID

#define IMPOSSIBLE_UNALIGNED_SIZE UINT64_C(0xefffffbd7fa092)
#define IMPOSSIBLE_HUGE_SIZE (UINT64_MAX & ~UINT64_C(511))

TEST_F(CacheMultiFile, SizeIsValid)
{
   set_counter(0);
   EXPECT_TRUE(disk_cache_size_is_valid(cache));

   set_counter(512);
   EXPECT_TRUE(disk_cache_size_is_valid(cache));

   set_counter(513);
   EXPECT_FALSE(disk_cache_size_is_valid(cache));

   set_counter(IMPOSSIBLE_UNALIGNED_SIZE);
   EXPECT_FALSE(disk_cache_size_is_valid(cache));

   set_counter(IMPOSSIBLE_HUGE_SIZE);
   EXPECT_FALSE(disk_cache_size_is_valid(cache));

   set_counter(0);
}

TEST_F(CacheMultiFile, RecomputeUnalignedSizeOnOpen)
{
   cache_key keys[3];
   put(1, 100, keys[0]);
   put(2, 5000, keys[1]);
   put(3, 20000, keys[2]);

   const uint64_t expected = size_on_disk();
   ASSERT_GT(expected, 0u);
   EXPECT_EQ(counter(), expected);

   reopen_with_counter(IMPOSSIBLE_UNALIGNED_SIZE);

   EXPECT_EQ(counter(), expected);
   for (unsigned i = 0; i < ARRAY_SIZE(keys); i++)
      EXPECT_TRUE(does_cache_contain(cache, keys[i]));
}

TEST_F(CacheMultiFile, RecomputeHugeSizeOnOpen)
{
   cache_key keys[2];
   put(1, 3000, keys[0]);
   put(2, 9000, keys[1]);

   const uint64_t expected = size_on_disk();
   ASSERT_GT(expected, 0u);

   reopen_with_counter(IMPOSSIBLE_HUGE_SIZE);

   EXPECT_EQ(counter(), expected);
   for (unsigned i = 0; i < ARRAY_SIZE(keys); i++)
      EXPECT_TRUE(does_cache_contain(cache, keys[i]));
}

TEST_F(CacheMultiFile, RecomputeEmptyCacheOnOpen)
{
   reopen_with_counter(IMPOSSIBLE_UNALIGNED_SIZE);

   EXPECT_EQ(counter(), 0u);

   cache_key key;
   put(1, 3000, key);
   EXPECT_TRUE(does_cache_contain(cache, key));
   EXPECT_EQ(counter(), size_on_disk());
}

TEST_F(CacheMultiFile, PutWithImpossibleSizeDoesNotEvict)
{
   cache_key first, second;
   put(1, 3000, first);
   ASSERT_TRUE(does_cache_contain(cache, first));

   set_counter(IMPOSSIBLE_UNALIGNED_SIZE);
   put(2, 3000, second);

   EXPECT_TRUE(does_cache_contain(cache, first));
   EXPECT_TRUE(does_cache_contain(cache, second));
   EXPECT_EQ(counter(), size_on_disk());
}

TEST_F(CacheMultiFile, PutWithOvercountedSizeRecomputes)
{
   cache_key first, second;
   put(1, 3000, first);
   ASSERT_TRUE(does_cache_contain(cache, first));

   set_counter(64 * 1024 * 1024);
   ASSERT_TRUE(disk_cache_size_is_valid(cache));
   put(2, 3000, second);

   EXPECT_TRUE(does_cache_contain(cache, second));
   EXPECT_EQ(counter(), size_on_disk());
}

TEST_F(CacheMultiFile, PutIntoEmptyCacheWithOvercountedSizeRecomputes)
{
   set_counter(64 * 1024 * 1024);
   ASSERT_TRUE(disk_cache_size_is_valid(cache));

   cache_key key;
   put(1, 3000, key);

   EXPECT_TRUE(does_cache_contain(cache, key));
   EXPECT_EQ(counter(), size_on_disk());
}

TEST_F(CacheMultiFile, EvictLruItemReportsWhetherItRemovedAFile)
{
   EXPECT_FALSE(disk_cache_evict_lru_item(cache));
   EXPECT_EQ(counter(), 0u);

   cache_key key;
   put(1, 3000, key);
   ASSERT_GT(counter(), 0u);

   EXPECT_TRUE(disk_cache_evict_lru_item(cache));
   EXPECT_FALSE(does_cache_contain(cache, key));
   EXPECT_EQ(counter(), 0u);

   EXPECT_FALSE(disk_cache_evict_lru_item(cache));
   EXPECT_EQ(counter(), 0u);
}

TEST_F(CacheMultiFile, RemoveKeepsSizeWhenUnlinkFails)
{
   if (geteuid() == 0)
      GTEST_SKIP() << "Directory permissions do not stop root from unlinking.";

   cache_key key;
   put(1, 3000, key);
   ASSERT_TRUE(does_cache_contain(cache, key));

   const uint64_t before = counter();
   ASSERT_GT(before, 0u);

   const std::string dir = item_dir(key);
   ASSERT_EQ(chmod(dir.c_str(), 0555), 0);
   disk_cache_remove(cache, key);
   EXPECT_EQ(chmod(dir.c_str(), 0755), 0);

   EXPECT_EQ(counter(), before);
   EXPECT_TRUE(does_cache_contain(cache, key));

   disk_cache_remove(cache, key);
   EXPECT_EQ(counter(), 0u);
   EXPECT_FALSE(does_cache_contain(cache, key));
}

TEST_F(CacheMultiFile, RecomputeRemovesStaleTmpFiles)
{
   cache_key key;
   put(1, 3000, key);

   const std::string dir = item_dir(key);
   const std::string stale = dir + "/stale.tmp";
   const std::string fresh = dir + "/fresh.tmp";
   const time_t now = time(NULL);

   create_file(stale, now - 2 * 60 * 60);
   create_file(fresh, now);

#ifdef HAVE_FLOCK
   const std::string locked = dir + "/locked.tmp";
   create_file(locked, now - 2 * 60 * 60);
   int locked_fd = open(locked.c_str(), O_WRONLY);
   ASSERT_NE(locked_fd, -1);
   ASSERT_EQ(flock(locked_fd, LOCK_EX | LOCK_NB), 0);
#endif

   const uint64_t expected = size_on_disk();
   reopen_with_counter(IMPOSSIBLE_UNALIGNED_SIZE);

   struct stat sb;
   EXPECT_EQ(stat(stale.c_str(), &sb), -1);
   EXPECT_EQ(errno, ENOENT);
   EXPECT_EQ(stat(fresh.c_str(), &sb), 0);
#ifdef HAVE_FLOCK
   EXPECT_EQ(stat(locked.c_str(), &sb), 0);
   close(locked_fd);
#endif

   EXPECT_EQ(counter(), expected);
   EXPECT_TRUE(does_cache_contain(cache, key));
}

TEST_F(CacheMultiFile, IndexMappingHasGuardPages)
{
   const size_t page_size = sysconf(_SC_PAGESIZE);

   ASSERT_NE(cache->index_reserve, nullptr);
   EXPECT_EQ(cache->index_reserve + page_size, cache->index_mmap);

   EXPECT_TRUE(is_readable_memory(cache->index_mmap));
   EXPECT_FALSE(is_readable_memory(cache->index_mmap - 1));
   EXPECT_FALSE(is_readable_memory(cache->index_reserve));
   EXPECT_FALSE(is_readable_memory(cache->index_reserve +
                                   cache->index_reserve_size - 1));
}

#endif /* ENABLE_SHADER_CACHE && !DETECT_OS_ANDROID */
