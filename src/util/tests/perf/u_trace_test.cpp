#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <list>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#ifdef _WIN32
#include <process.h>
#define test_getpid _getpid
#else
#include <unistd.h>
#define test_getpid getpid
#endif

#include "c11/threads.h"
#include "util/perf/u_trace.h"
/* For struct u_tracepoint and u_trace_append(), as generated tracepoints do. */
#define __NEEDS_TRACE_PRIV
#include "util/perf/u_trace_priv.h"
#include "util/os_misc.h"

#if U_TRACE_HAS_COMPRESS
#include <zlib.h>
#endif

#define NUM_DEBUG_TEST_THREAD 8

static const char *trace_file_path =
   "tracefile_for_test-b5ba5a0c-6ed1-4901-a38d-755991182663";

static std::string
trace_file_with(const char *suffix)
{
   return std::string(trace_file_path) + suffix;
}

/* u_trace reads its MESA_GPU_TRACE* options once, so the state is reset
 * around each test to let every case set its own.
 */
class UtilPerfTraceTest : public ::testing::Test {
protected:
   void SetUp() override
   {
      u_trace_state_reset();
   }

   void TearDown() override
   {
      u_trace_state_reset();
      remove(trace_file_path);
   }
};

static void
init_context(struct u_trace_context *ctx)
{
   memset(ctx, 0, sizeof(*ctx));
   u_trace_context_init(ctx, NULL, 8, 0, NULL, NULL, NULL,
                        NULL, NULL, NULL, NULL);
}

/* Opens and closes one frame with an empty batch, which needs no timestamp
 * callbacks, and waits for it to be printed.
 */
static void
process_frame(struct u_trace_context *ctx)
{
   struct u_trace ut;
   u_trace_init(&ut, ctx);
   u_trace_flush(&ut, NULL, 0, false);
   u_trace_fini(&ut);
   u_trace_context_process(ctx, true);
   util_queue_finish(&ctx->queue);
}

static std::string
read_file(const char *path)
{
   std::string contents;
   FILE *f = fopen(path, "rb");
   if (!f)
      return contents;

   char buf[256];
   size_t n;
   while ((n = fread(buf, 1, sizeof(buf), f)) > 0)
      contents.append(buf, n);
   fclose(f);
   return contents;
}

#if U_TRACE_HAS_COMPRESS
/* complete is cleared if the stream ends without its trailer, i.e. it was
 * never closed.
 */
static std::string
read_gz_file(const char *path, bool *complete)
{
   std::string contents;
   *complete = false;
   gzFile f = gzopen(path, "rb");
   if (!f)
      return contents;

   char buf[256];
   int n;
   while ((n = gzread(f, buf, sizeof(buf))) > 0)
      contents.append(buf, n);

   int err;
   gzerror(f, &err);
   *complete = n == 0 && err == Z_OK;
   gzclose(f);
   return contents;
}
#endif

static int
count_lines_starting_with(const std::string &contents, const std::string &prefix)
{
   int count = 0;
   for (size_t pos = 0; pos < contents.size();) {
      if (contents.compare(pos, prefix.size(), prefix) == 0)
         count++;
      size_t newline = contents.find('\n', pos);
      if (newline == std::string::npos)
         break;
      pos = newline + 1;
   }
   return count;
}

static int
test_thread(void *_state)
{
   struct u_trace_context ctx;
   init_context(&ctx);
   u_trace_context_fini(&ctx);

   return 0;
}

TEST_F(UtilPerfTraceTest, Multithread)
{
   thrd_t threads[NUM_DEBUG_TEST_THREAD];
   os_set_option("MESA_GPU_TRACEFILE", trace_file_path, true);
   /* No "%i", so all threads share one file; exercises shared_out_mtx. */
   os_set_option("MESA_GPU_TRACES", "print_csv", true);

   for (unsigned i = 0; i < NUM_DEBUG_TEST_THREAD; i++) {
        thrd_create(&threads[i], test_thread, NULL);
   }
   for (unsigned i = 0; i < NUM_DEBUG_TEST_THREAD; i++) {
      int ret;
      thrd_join(threads[i], &ret);
   }
   u_trace_state_reset();

   /* Each thread writes one known header line and one blank line; check
    * none got torn or merged by concurrent writers.
    */
   std::string out = read_file(trace_file_path);
   EXPECT_EQ(count_lines_starting_with(out, "frame,batch,time_ns,event,"),
             NUM_DEBUG_TEST_THREAD) << out;
   int newlines = 0;
   for (char c : out) {
      if (c == '\n')
         newlines++;
   }
   EXPECT_EQ(newlines, 2 * NUM_DEBUG_TEST_THREAD) << out;
}

TEST_F(UtilPerfTraceTest, DisabledHasNoQueue)
{
   os_set_option("MESA_GPU_TRACES", NULL, true);

   struct u_trace_context ctx;
   init_context(&ctx);
   EXPECT_FALSE(util_queue_is_initialized(&ctx.queue));

   struct u_trace ut;
   u_trace_init(&ut, &ctx);
   u_trace_flush(&ut, NULL, 0, false);
   u_trace_fini(&ut);
   u_trace_context_process(&ctx, true);
   EXPECT_FALSE(util_queue_is_initialized(&ctx.queue));

   u_trace_context_fini(&ctx);
}

TEST_F(UtilPerfTraceTest, EndOfFrameOnce)
{
   os_set_option("MESA_GPU_TRACEFILE", trace_file_path, true);
   os_set_option("MESA_GPU_TRACES", "print", true);

   struct u_trace_context ctx;
   init_context(&ctx);
   process_frame(&ctx);
   u_trace_context_fini(&ctx);
   u_trace_state_reset();

   std::string out = read_file(trace_file_path);
   EXPECT_EQ(count_lines_starting_with(out, "END OF FRAME 0 (ctx 0)"), 1) << out;
}

TEST_F(UtilPerfTraceTest, PerContextFiles)
{
   std::string path_template = trace_file_with("-%i");
   os_set_option("MESA_GPU_TRACEFILE", path_template.c_str(), true);
   os_set_option("MESA_GPU_TRACES", "print", true);

   struct u_trace_context ctx[2];
   for (unsigned i = 0; i < ARRAY_SIZE(ctx); i++) {
      init_context(&ctx[i]);
      process_frame(&ctx[i]);
      u_trace_context_fini(&ctx[i]);
   }
   u_trace_state_reset();

   for (unsigned i = 0; i < ARRAY_SIZE(ctx); i++) {
      std::string path = trace_file_with(("-" + std::to_string(i)).c_str());
      std::string out = read_file(path.c_str());
      std::string marker = "END OF FRAME 0 (ctx " + std::to_string(i) + ")";
      EXPECT_EQ(count_lines_starting_with(out, marker), 1) << out;
      remove(path.c_str());
   }
}

TEST_F(UtilPerfTraceTest, PerProcessFile)
{
   std::string path_template = trace_file_with("-%p");
   os_set_option("MESA_GPU_TRACEFILE", path_template.c_str(), true);
   os_set_option("MESA_GPU_TRACES", "print", true);

   struct u_trace_context ctx;
   init_context(&ctx);
   process_frame(&ctx);
   u_trace_context_fini(&ctx);
   u_trace_state_reset();

   std::string path =
      trace_file_with(("-" + std::to_string(test_getpid())).c_str());
   std::string out = read_file(path.c_str());
   EXPECT_EQ(count_lines_starting_with(out, "END OF FRAME 0 (ctx 0)"), 1) << out;
   remove(path.c_str());
}

#if U_TRACE_HAS_COMPRESS
TEST_F(UtilPerfTraceTest, GzipPerContextFiles)
{
   std::string path_template = trace_file_with("-%i.gz");
   os_set_option("MESA_GPU_TRACEFILE", path_template.c_str(), true);
   os_set_option("MESA_GPU_TRACES", "print", true);

   struct u_trace_context ctx[2];
   for (unsigned i = 0; i < ARRAY_SIZE(ctx); i++) {
      init_context(&ctx[i]);
      process_frame(&ctx[i]);
      u_trace_context_fini(&ctx[i]);
   }
   u_trace_state_reset();

   for (unsigned i = 0; i < ARRAY_SIZE(ctx); i++) {
      std::string path = trace_file_with(("-" + std::to_string(i) + ".gz").c_str());
      bool complete;
      std::string out = read_gz_file(path.c_str(), &complete);
      EXPECT_TRUE(complete) << path;
      std::string marker = "END OF FRAME 0 (ctx " + std::to_string(i) + ")";
      EXPECT_EQ(count_lines_starting_with(out, marker), 1) << out;
      remove(path.c_str());
   }
}

TEST_F(UtilPerfTraceTest, GzipSharedFile)
{
   std::string path = trace_file_with(".gz");
   os_set_option("MESA_GPU_TRACEFILE", path.c_str(), true);
   os_set_option("MESA_GPU_TRACES", "print", true);

   struct u_trace_context ctx[2];
   for (unsigned i = 0; i < ARRAY_SIZE(ctx); i++) {
      init_context(&ctx[i]);
      process_frame(&ctx[i]);
      u_trace_context_fini(&ctx[i]);
   }
   /* The shared stream is only closed here, with the process-wide state. */
   u_trace_state_reset();

   bool complete;
   std::string out = read_gz_file(path.c_str(), &complete);
   EXPECT_TRUE(complete);
   for (unsigned i = 0; i < ARRAY_SIZE(ctx); i++) {
      std::string marker = "END OF FRAME 0 (ctx " + std::to_string(i) + ")";
      EXPECT_EQ(count_lines_starting_with(out, marker), 1) << out;
   }
   remove(path.c_str());
}
#endif

/* CPU stand-ins for a driver's timestamp buffers, reached from the u_trace
 * callbacks through the context's pctx. A deleted buffer is poisoned rather
 * than freed, so a clone still pointing into its source reads garbage.
 */
struct fake_timestamps {
   std::list<std::vector<uint64_t>> buffers;
   uint64_t next_ts = 0;
   bool reuse_next = false;
   unsigned copies = 0;
   uint64_t bytes_copied = 0;
   std::vector<uint64_t> read_back;

   static fake_timestamps *of(struct u_trace_context *utctx)
   {
      return (fake_timestamps *) utctx->pctx;
   }

   static void *create_buffer(struct u_trace_context *utctx, uint64_t size_B)
   {
      return of(utctx)->buffers.emplace_back(size_B / sizeof(uint64_t)).data();
   }

   static void delete_buffer(struct u_trace_context *utctx, void *buffer)
   {
      for (std::vector<uint64_t> &b : of(utctx)->buffers) {
         if (b.data() == buffer)
            std::fill(b.begin(), b.end(), UINT64_MAX);
      }
   }

   static bool record_ts(struct u_trace *ut, void *cs, void *timestamps,
                         uint64_t offset_B, uint32_t flags)
   {
      fake_timestamps *f = of(ut->utctx);
      if (f->reuse_next)
         return false;

      ((uint64_t *) timestamps)[offset_B / sizeof(uint64_t)] = ++f->next_ts;
      return true;
   }

   static uint64_t read_ts(struct u_trace_context *utctx, void *timestamps,
                           uint64_t offset_B, uint32_t flags, void *flush_data)
   {
      uint64_t ts = ((uint64_t *) timestamps)[offset_B / sizeof(uint64_t)];
      of(utctx)->read_back.push_back(ts);
      return ts;
   }

   static void copy_buffer(struct u_trace_context *utctx, void *cmdstream,
                           void *ts_from, uint64_t from_offset, void *ts_to,
                           uint64_t to_offset, uint64_t size_B)
   {
      memcpy((char *) ts_to + to_offset, (char *) ts_from + from_offset, size_B);
      of(utctx)->copies++;
      of(utctx)->bytes_copied += size_B;
   }
};

static const struct u_tracepoint clone_test_tp = { "clone_test" };

class UtilPerfTraceCloneTest : public ::testing::Test {
protected:
   fake_timestamps fake;
   struct u_trace_context ctx = {};
   struct u_trace src;

   void SetUp() override
   {
      u_trace_state_reset();
      u_trace_context_init(&ctx, &fake, sizeof(uint64_t), 0,
                           fake_timestamps::create_buffer,
                           fake_timestamps::delete_buffer,
                           fake_timestamps::record_ts,
                           fake_timestamps::read_ts, NULL, NULL, NULL);
      u_trace_init(&src, &ctx);
   }

   void TearDown() override
   {
      u_trace_fini(&src);
      u_trace_context_fini(&ctx);
      u_trace_state_reset();
   }

   void append(struct u_trace *ut, unsigned count = 1, bool reuse = false)
   {
      fake.reuse_next = reuse;
      for (unsigned i = 0; i < count; i++)
         u_trace_append(ut, NULL, &clone_test_tp);
      fake.reuse_next = false;
   }

   /* Clones [begin, end) of the source after `own` events of the
    * destination's own, deletes the source, and returns the timestamps the
    * clone reads back.
    */
   std::vector<uint64_t> clone(struct u_trace_iterator begin,
                               struct u_trace_iterator end, unsigned own)
   {
      struct u_trace dst;
      u_trace_init(&dst, &ctx);
      append(&dst, own);

      fake.copies = 0;
      fake.bytes_copied = 0;
      u_trace_clone_append(begin, end, &dst, NULL, fake_timestamps::copy_buffer);
      EXPECT_EQ(fake.copies, u_trace_clone_append_copy_count(begin, end));

      u_trace_fini(&src);
      u_trace_init(&src, &ctx);

      fake.read_back.clear();
      u_trace_flush(&dst, NULL, 0, false);
      u_trace_context_process(&ctx, true);
      util_queue_finish(&ctx.queue);
      u_trace_fini(&dst);
      return fake.read_back;
   }
};

TEST_F(UtilPerfTraceCloneTest, CopiesOnlyTheClonedRange)
{
   append(&src, 3);
   struct u_trace_iterator begin = u_trace_end_iterator(&src);
   append(&src, 5);
   struct u_trace_iterator end = u_trace_end_iterator(&src);
   append(&src, 2);

   /* The destination's own two events are recorded after the source's ten. */
   EXPECT_EQ(clone(begin, end, 2),
             (std::vector<uint64_t>{ 11, 12, 4, 5, 6, 7, 8 }));
   EXPECT_EQ(fake.bytes_copied, 5 * sizeof(uint64_t));
}

TEST_F(UtilPerfTraceCloneTest, RangeAcrossTimestampBuffers)
{
   append(&src);
   uint64_t per_buffer = fake.buffers.front().size();

   /* Two events at the end of the first timestamp buffer, two in a second. */
   append(&src, per_buffer - 3);
   struct u_trace_iterator begin = u_trace_end_iterator(&src);
   append(&src, 4);
   struct u_trace_iterator end = u_trace_end_iterator(&src);
   ASSERT_EQ(fake.buffers.size(), 2u);

   EXPECT_EQ(clone(begin, end, 0),
             (std::vector<uint64_t>{ per_buffer - 1, per_buffer,
                                     per_buffer + 1, per_buffer + 2 }));
   EXPECT_EQ(fake.copies, 2u);
   EXPECT_EQ(fake.bytes_copied, 4 * sizeof(uint64_t));
}

TEST_F(UtilPerfTraceCloneTest, ReusedTimestampFromBeforeTheRange)
{
   append(&src);
   struct u_trace_iterator begin = u_trace_end_iterator(&src);
   append(&src, 1, true);
   append(&src);
   append(&src, 1, true);
   struct u_trace_iterator end = u_trace_end_iterator(&src);

   /* The first cloned event reuses a timestamp recorded before the range.
    * Read-back skips an event whose timestamp is the previous event's.
    */
   EXPECT_EQ(clone(begin, end, 0), (std::vector<uint64_t>{ 1, 2 }));
}
