import os

import pytest

from engine_sim import run_random_case, run_random_case_block_256

NUM_CASES = int(os.environ.get("NANOVLLM_SIM_CASES", "300"))
# (async_scheduling, num_speculative_tokens, draft_model, adaptive: drafts switched off for larger batches,
#  acceptance: drafts switched off while the measured acceptance rate is low, per_sequence: a number of drafts per sequence, eagle: an EAGLE-3 head)
MODES = {"sync": (False, 0, False, False, False), "async": (True, 0, False, False, False), "spec1": (False, 1, False, False, False),
         "spec4": (False, 4, False, False, False), "draft1": (False, 1, True, False, False), "draft4": (False, 4, True, False, False),
         "async_draft1": (True, 1, True, False, False), "async_draft4": (True, 4, True, False, False),
         "spec4_adaptive": (False, 4, False, True, False), "draft4_adaptive": (False, 4, True, True, False), "async_draft4_adaptive": (True, 4, True, True, False),
         "spec4_acc": (False, 4, False, False, True), "draft4_acc": (False, 4, True, False, True), "async_draft4_acc": (True, 4, True, False, True),
         "spec4_perseq": (False, 4, False, False, True, True), "draft4_perseq": (False, 4, True, False, True, True), "async_draft4_perseq": (True, 4, True, False, True, True),
         "spec2_perseq": (False, 2, False, False, True, True), "draft4_adaptive_perseq": (False, 4, True, True, True, True),
         # eagle: an EAGLE-3 head drafts (one position behind the target, from its hidden states) instead of a draft model
         "eagle1": (False, 1, False, False, False, False, True), "eagle4": (False, 4, False, False, False, False, True),
         "async_eagle4": (True, 4, False, False, False, False, True), "eagle4_adaptive": (False, 4, False, True, False, False, True),
         "async_eagle4_adaptive": (True, 4, False, True, False, False, True), "eagle4_acc": (False, 4, False, False, True, False, True),
         "eagle4_perseq": (False, 4, False, False, True, True, True), "async_eagle4_perseq": (True, 4, False, False, True, True, True)}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("seed", range(NUM_CASES))
def test_random_engine(seed, mode):
    run_random_case(seed, *MODES[mode])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("seed", range(max(1, NUM_CASES // 10)))
def test_random_engine_block_256(seed, mode):
    run_random_case_block_256(seed, *MODES[mode])
