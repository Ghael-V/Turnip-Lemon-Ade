# PowerVR

PowerVR is an open source Vulkan driver for Imagination Technologies PowerVR
GPUs, starting with those based on the Rogue architecture.

Additional documentation can be found on our [Mesa docs page](https://docs.mesa3d.org/drivers/powervr.html).

## Contribution policy

Further to the [upstream Mesa policy](https://docs.mesa3d.org/submittingpatches.html#expectations-on-contributors),
we politely decline LLM-generated code contributions as part of our commitment
to continue improving the quality and maintainability of our codebase while
growing a knowledgeable team and community.

<details>
<summary>For additional context:</summary>

As things stand, our open-source team and community are relatively small, with
a number of newer members and outside contributors not yet having had
significant time on, or experience with, our hardware and/or the open-source
drivers. Reviewing can at times fall onto an even smaller subset of people,
which inherently limits our throughput, and we have found that LLM-generated
changes often take a disproportionate amount of time to address and respond to
(especially when compared to the time needed to generate said changes).

Internally, our approach has been to largely discourage using LLMs to generate
code for our driver(s) and compiler(s), especially for newer members of the
team. This is largely because although LLMs seem capable of implementing
features and fixes that technically function, the resulting quality and
maintainability of the code has frequently left a lot to be desired. These
shortcomings can be difficult for those without a certain level of familiarity
with the hardware and codebase to detect, which risks adding technical debt and
introducing cases where important code "works" but is poorly understood and
difficult to change.

The process of using an LLM to implement changes without the contributor
independently having a strong understanding of the underlying components also
inherently hinders learning, which exacerbates the widely discussed junior <->
senior engineer development problem, and further delays the growth of our
experienced reviewer "pool".

Our driver, compiler, and other components are currently going through
significant changes, including (but not limited to) adding support for another
GPU architecture, switching more code over to using vk-meta & vulkan/runtime,
and addressing long-standing technical debt. These activities, as well as having
a knowledgeable team to work on them, are crucial to the long-term health of our
codebase.

As of such, we have decided to effectively extend our internal approach to LLMs
with this policy, and thus politely decline LLM-generated contributions for the
time being.

</details>

This policy will continue to evolve over time.
