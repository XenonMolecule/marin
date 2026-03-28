# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: validate llama.cpp CPU inference pipeline with 2 workers and 8 prompts.

This is a minimal end-to-end test of the llama.cpp + Zephyr integration. It:
  1. Generates 8 synthetic HTML documents (~1k tokens each) and writes them to GCS
  2. Runs inference via llama-server with 2 CPU workers
  3. Writes results as jsonl.gz

If this succeeds, the full-scale rephraser_cooldown_cpu.py pipeline is feasible.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-east5-a --no_wait \\
        -- python experiments/rephraser/test_llamacpp_smoke.py
"""

import gzip
import json
import logging

import fsspec

from fray.cluster import ResourceConfig

from marin.execution.remote import remote
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.generation.build_llamacpp import BuildLlamaCppConfig, build_llamacpp
from marin.generation.inference_llamacpp import LlamaCppInferenceConfig, run_inference_llamacpp

from experiments.rephraser.rephraser_cooldown import spec_hash
from experiments.rephraser.rephraser_cooldown_cpu import DownloadGGUFConfig, download_gguf_to_gcs

logger = logging.getLogger(__name__)

# Reuse the already-cached build step (same name = cache hit on us-east5-a)
build_llamacpp_step = ExecutorStep(
    name="tools/llamacpp-build-v3",
    description="Build llama.cpp from source (cached).",
    fn=build_llamacpp,
    config=BuildLlamaCppConfig(output_path=this_output_path()),
)

# GGUF model — download once to GCS as an ExecutorStep, then workers fetch
# from GCS (fast, in-region) instead of HuggingFace.
GGUF_SOURCE_URL = (
    "https://huggingface.co/MichaelR207/"
    "qwen3-1.7b-rephraser-sft-mid-ckpt5000-Q4_K_M-GGUF/resolve/main/"
    "qwen3-1.7b-rephraser-sft-mid-ckpt5000-q4_k_m.gguf"
)
GGUF_FILENAME = "qwen3-1.7b-rephraser-sft-mid-ckpt5000-q4_k_m.gguf"

# Step name includes a hash of the source URL so changing models invalidates the cache.
_gguf_url_hash = spec_hash(GGUF_SOURCE_URL)[:8]

download_gguf_step = ExecutorStep(
    name=f"models/gguf-{_gguf_url_hash}",
    description="Download GGUF model from HuggingFace and cache in GCS.",
    fn=download_gguf_to_gcs,
    config=DownloadGGUFConfig(
        source_url=GGUF_SOURCE_URL,
        output_path=this_output_path(),
        filename=GGUF_FILENAME,
    ),
)

# ---------------------------------------------------------------------------
# Step 1: Generate synthetic test data
# ---------------------------------------------------------------------------

# 8 short HTML pages (~1k tokens each ≈ ~4k chars). Just enough to exercise
# the full pipeline: download model, start server, process, write output.
TEST_DOCUMENTS = [
    {
        "html": (
            "<html><head><title>Climate Change Report 2024</title></head><body>"
            "<h1>Global Climate Summary</h1>"
            "<p>Average global temperatures rose by 1.2 degrees Celsius above pre-industrial "
            "levels in 2024. The Arctic experienced record low sea ice extent in September, "
            "with coverage dropping to 3.4 million square kilometers. Major hurricanes in the "
            "Atlantic basin numbered seven, with three making landfall. Coral bleaching events "
            "affected 60 percent of tropical reefs worldwide. Carbon dioxide concentrations "
            "reached 425 parts per million at the Mauna Loa observatory. Renewable energy "
            "installations grew by 35 percent year over year, led by solar photovoltaics. "
            "Electric vehicle sales exceeded 20 million units globally. Despite progress in "
            "clean energy, total greenhouse gas emissions increased by 0.8 percent. Sea level "
            "rise accelerated to 4.5 millimeters per year, threatening coastal communities. "
            "Permafrost thaw in Siberia released significant methane plumes detected by satellite. "
            "The European Union implemented its carbon border adjustment mechanism. China "
            "commissioned 50 gigawatts of new nuclear capacity. India surpassed its 500 gigawatt "
            "renewable energy target ahead of schedule. Deforestation in the Amazon declined by "
            "30 percent following enforcement measures. The global carbon market reached 950 "
            "billion dollars in traded value. Climate adaptation funding doubled to 80 billion "
            "dollars annually. Small island developing states reported accelerating coastal "
            "erosion. Atmospheric methane levels plateaued for the first time in a decade. "
            "New satellite constellations improved climate monitoring resolution tenfold.</p>"
            "</body></html>"
        )
    },
    {
        "html": (
            "<html><head><title>Machine Learning Survey</title></head><body>"
            "<h1>Recent Advances in Deep Learning</h1>"
            "<p>Transformer architectures continued to dominate natural language processing in "
            "2024. Large language models scaled to over one trillion parameters. Mixture of "
            "experts models showed significant efficiency gains at inference time. Retrieval "
            "augmented generation became the standard approach for knowledge-intensive tasks. "
            "Vision transformers achieved state of the art on ImageNet with 92 percent top-1 "
            "accuracy. Multimodal models unified text, image, audio, and video understanding. "
            "Diffusion models generated photorealistic images and videos at unprecedented quality. "
            "Reinforcement learning from human feedback improved model alignment with human "
            "preferences. Constitutional AI methods reduced harmful outputs by 85 percent in "
            "benchmarks. Distillation techniques compressed 70 billion parameter models to 7 "
            "billion with minimal quality loss. Quantization to 4-bit precision enabled running "
            "large models on consumer hardware. Sparse attention mechanisms reduced memory usage "
            "by 4x for long context processing. Chain of thought prompting improved mathematical "
            "reasoning accuracy by 40 percent. Tool use capabilities enabled language models to "
            "interact with external APIs and databases. Code generation models passed 80 percent "
            "of competitive programming challenges. Autonomous agents built on language models "
            "demonstrated complex multi-step planning. Federated learning enabled privacy "
            "preserving model training across institutions. Neural architecture search discovered "
            "novel efficient architectures automatically.</p>"
            "</body></html>"
        )
    },
    {
        "html": (
            "<html><head><title>Ocean Conservation</title></head><body>"
            "<h1>Marine Ecosystem Protection Status</h1>"
            "<p>Global marine protected areas expanded to cover 12 percent of ocean surface. "
            "Deep sea mining moratorium extended through 2030 by international agreement. "
            "Plastic pollution in the Pacific garbage patch decreased by 15 percent following "
            "cleanup operations. Whale populations in the North Atlantic showed recovery with "
            "humpback numbers reaching 25000. Coral restoration projects in the Great Barrier "
            "Reef showed 40 percent survival rates for transplanted colonies. Sustainable "
            "fishing practices adoption reached 60 percent of global commercial fleets. "
            "Mangrove restoration projects sequestered an estimated 3 million tonnes of carbon. "
            "Ocean acidification monitoring expanded to 500 stations worldwide. Seagrass meadows "
            "were recognized as critical carbon sinks storing 10 percent of ocean carbon. "
            "Marine heatwave events doubled in frequency compared to the 1990s baseline. "
            "International shipping emissions regulations reduced sulfur output by 70 percent. "
            "Offshore wind farms provided habitat benefits for marine species. Acoustic monitoring "
            "networks tracked whale migration patterns across entire ocean basins. Illegal fishing "
            "detection improved through satellite surveillance and AI analysis. Microplastic "
            "concentrations in deep ocean sediments exceeded surface water levels. New species "
            "discovery rate in deep sea environments averaged 50 per year. Kelp forest restoration "
            "along temperate coastlines showed promising ecosystem recovery signals.</p>"
            "</body></html>"
        )
    },
    {
        "html": (
            "<html><head><title>Urban Planning Report</title></head><body>"
            "<h1>Smart City Infrastructure Development</h1>"
            "<p>Cities worldwide invested 200 billion dollars in smart infrastructure upgrades. "
            "Autonomous public transit systems operated in 30 metropolitan areas globally. "
            "Building energy efficiency codes reduced commercial energy consumption by 25 percent. "
            "Urban tree canopy coverage targets of 40 percent were adopted by 100 cities. "
            "Pedestrian zones expanded by 50 percent in European city centers. Bike sharing "
            "programs served 500 million rides annually across participating cities. Mixed use "
            "zoning reforms reduced average commute times by 15 minutes. Affordable housing "
            "production increased through modular construction techniques. Smart water management "
            "systems reduced municipal water waste by 30 percent. Air quality monitoring networks "
            "provided real time data at neighborhood resolution. Green roof installations covered "
            "50 square kilometers of urban rooftop space. Underground utility modernization "
            "eliminated 10000 kilometers of aging lead pipes. Electric bus fleets comprised "
            "40 percent of new public transit vehicle purchases. Urban agriculture initiatives "
            "produced 5 percent of city food consumption locally. Noise pollution regulations "
            "established quiet zones around hospitals and schools. Digital twin technology enabled "
            "simulation of infrastructure changes before implementation. Community solar programs "
            "provided renewable energy access to apartment residents. Stormwater management "
            "improvements prevented flooding in 200 previously vulnerable neighborhoods.</p>"
            "</body></html>"
        )
    },
    {
        "html": (
            "<html><head><title>Medical Research Update</title></head><body>"
            "<h1>Breakthroughs in Biomedical Science</h1>"
            "<p>mRNA vaccine technology expanded to target influenza and malaria with promising "
            "trial results. CRISPR gene editing received approval for sickle cell disease "
            "treatment in 15 countries. Artificial intelligence diagnostics matched radiologist "
            "accuracy in detecting lung cancer from CT scans. Wearable health monitors tracked "
            "continuous glucose and blood pressure for 100 million users. Telemedicine visits "
            "accounted for 35 percent of primary care consultations. Organ on chip technology "
            "reduced animal testing requirements by 50 percent in drug development. Xenotransplant "
            "pig kidneys functioned for over one year in human recipients. Microbiome therapies "
            "showed efficacy in treating inflammatory bowel disease. Liquid biopsy blood tests "
            "detected early stage cancers with 90 percent sensitivity. Brain computer interfaces "
            "restored speech capability in paralyzed patients. Digital therapeutics received "
            "prescriptive authorization for insomnia and substance abuse disorders. Personalized "
            "cancer vaccines demonstrated tumor regression in 60 percent of melanoma cases. "
            "Antibiotic resistance surveillance identified 15 new concerning resistance patterns. "
            "Long acting injectable medications improved HIV treatment adherence to 95 percent. "
            "Robotic surgery systems performed 2 million minimally invasive procedures annually. "
            "Stem cell therapies regenerated cardiac tissue in post heart attack patients. "
            "Mental health screening tools using natural language processing identified at risk "
            "individuals with 80 percent accuracy.</p>"
            "</body></html>"
        )
    },
    {
        "html": (
            "<html><head><title>Space Exploration Summary</title></head><body>"
            "<h1>Space Mission Highlights</h1>"
            "<p>The Artemis program completed its second crewed lunar landing mission. SpaceX "
            "Starship achieved full reusability with 20 consecutive successful booster catches. "
            "The James Webb Space Telescope discovered atmospheric signatures on 5 rocky "
            "exoplanets in habitable zones. Mars sample return mission collected 30 sealed tubes "
            "from the Jezero crater. The International Space Station operations extended through "
            "2030 with commercial module additions. Satellite internet constellations reached "
            "50000 active spacecraft in low Earth orbit. Space debris mitigation removed 100 "
            "defunct satellites through active capture missions. Private space stations from "
            "three companies began construction in low Earth orbit. The European Space Agency "
            "launched its first gravitational wave observatory in space. China completed assembly "
            "of its third space station module with permanent crew rotation. India successfully "
            "landed its second rover on the lunar south pole. Nuclear thermal propulsion testing "
            "demonstrated 900 seconds of specific impulse. Asteroid mining companies completed "
            "prospecting missions to three near earth asteroids. Space tourism flights carried "
            "200 private citizens above the Karman line. Solar sail technology propelled a "
            "spacecraft to Jupiter in record time. The Square Kilometre Array began scientific "
            "observations detecting fast radio bursts at unprecedented rates.</p>"
            "</body></html>"
        )
    },
    {
        "html": (
            "<html><head><title>Agricultural Innovation</title></head><body>"
            "<h1>Sustainable Farming Technology Report</h1>"
            "<p>Precision agriculture adoption reached 70 percent of large scale farms globally. "
            "Vertical farming operations produced 5 million tonnes of leafy greens annually. "
            "Drought resistant crop varieties increased yields by 25 percent in arid regions. "
            "Robotic harvesting systems reduced labor requirements by 40 percent for fruit crops. "
            "Soil carbon sequestration programs enrolled 100 million hectares of farmland. "
            "Biological pest control methods replaced chemical pesticides on 30 percent of acreage. "
            "Satellite based crop monitoring provided weekly health assessments for all major "
            "growing regions. Gene edited crops with enhanced nutritional profiles received "
            "regulatory approval in 20 countries. Regenerative grazing practices restored 50 "
            "million hectares of degraded grassland. Agricultural drones applied targeted "
            "treatments reducing herbicide use by 60 percent. Indoor mushroom farming scaled to "
            "industrial production levels using agricultural waste substrates. Aquaponics systems "
            "combined fish farming with vegetable production in urban facilities. Cover crop "
            "adoption improved soil health metrics across 80 million hectares. Blockchain based "
            "supply chain tracking ensured food provenance from farm to consumer. Climate smart "
            "irrigation systems reduced water consumption by 40 percent while maintaining yields. "
            "Alternative protein production from fermentation reached 2 million tonnes annually. "
            "Pollinator protection programs increased wild bee populations by 20 percent in "
            "participating regions.</p>"
            "</body></html>"
        )
    },
    {
        "html": (
            "<html><head><title>Education Technology Review</title></head><body>"
            "<h1>Digital Learning Transformation</h1>"
            "<p>Adaptive learning platforms served 500 million students across 100 countries. "
            "AI tutoring systems demonstrated learning gains equivalent to one on one human "
            "tutoring in mathematics. Virtual reality classrooms enabled immersive science "
            "education for remote schools. Open educational resources covered 80 percent of "
            "undergraduate curricula in standardized formats. Automated essay grading reached "
            "95 percent agreement with human evaluators. Competency based credentials gained "
            "acceptance at 500 major employers worldwide. Language learning applications achieved "
            "conversational fluency outcomes in 6 months for motivated users. Collaborative "
            "online laboratories allowed hands on science experiments through remote robotic "
            "interfaces. Learning analytics dashboards provided teachers with real time student "
            "engagement metrics. Gamification elements increased course completion rates by "
            "35 percent in online programs. Accessibility tools including automatic captioning "
            "and screen readers served 50 million learners with disabilities. Peer assessment "
            "platforms scaled feedback to classes of 10000 students. Digital credentialing "
            "systems using verifiable certificates reduced credential fraud by 90 percent. "
            "Microlearning modules of 5 to 10 minutes showed higher retention than traditional "
            "lecture formats. Teacher professional development moved online with 70 percent of "
            "continuing education delivered digitally. Cross institutional course sharing "
            "agreements expanded student access to specialized programs.</p>"
            "</body></html>"
        )
    },
]

# A simple extraction spec for the smoke test
SMOKE_SPEC = "Extract the main topic, key statistics (as bullet points), and a one-paragraph summary."

SYSTEM_MESSAGE = (
    "Your input fields are:\n"
    "1. `html` (str): \n"
    "2. `extraction_spec` (str):\n\n"
    "Your output fields are:\n"
    "1. `text` (str):\n\n"
    "All interactions will be structured in the following way, with the appropriate values filled in.\n\n"
    "[[ ## html ## ]]\n{html}\n\n[[ ## extraction_spec ## ]]\n{extraction_spec}\n\n"
    "[[ ## text ## ]]\n{text}\n\n[[ ## completed ## ]]"
)

USER_TEMPLATE = (
    "[[ ## html ## ]]\n{example}\n\n"
    f"[[ ## extraction_spec ## ]]\n{SMOKE_SPEC}\n\n"
    "Respond with the corresponding output fields, "
    "starting with [[ ## text ## ]], and then ending with [[ ## completed ## ]]."
)


# ---------------------------------------------------------------------------
# Step 1: Write test data to GCS
# ---------------------------------------------------------------------------


def write_test_data(config: dict) -> None:
    """Write synthetic test documents to GCS as a single jsonl.gz file."""
    output_path = config["output_path"]
    data_path = f"{output_path}/test_data.jsonl.gz"

    logger.info("Writing %d test documents to %s", len(TEST_DOCUMENTS), data_path)
    with fsspec.open(data_path, "wb") as f:
        with gzip.open(f, "wt", encoding="utf-8") as gz:
            for doc in TEST_DOCUMENTS:
                gz.write(json.dumps(doc) + "\n")

    logger.info("Test data written: %s", data_path)


write_test_data_step = ExecutorStep(
    name="test/llamacpp_smoke_data",
    description="Write 8 synthetic HTML documents for smoke testing.",
    fn=write_test_data,
    config={"output_path": this_output_path()},
)

# ---------------------------------------------------------------------------
# Step 2: Run inference with 2 workers
# ---------------------------------------------------------------------------
inference_step = ExecutorStep(
    name="test/llamacpp_smoke_inference_v2",
    description="Smoke test: llama.cpp CPU inference with 2 workers on 8 documents.",
    fn=remote(run_inference_llamacpp, pip_dependency_groups=["cpu"]),
    config=LlamaCppInferenceConfig(
        input_path=write_test_data_step / "*.jsonl.gz",
        output_path=this_output_path(),
        gguf_model_path=download_gguf_step / GGUF_FILENAME,
        llamacpp_binary_path=output_path_of(build_llamacpp_step),
        input_format="jsonl.gz",
        output_format="jsonl.gz",
        system_message=SYSTEM_MESSAGE,
        template=USER_TEMPLATE,
        prompt_column="html",
        max_tokens=1024,
        temperature=0.0,
        # Small scale: 2 workers, 8 threads each
        threads_per_worker=8,
        num_workers=2,
        cpu_per_worker=8,
        ram_per_worker="8g",
        context_length=8192,
        records_per_shard=4,
        server_startup_timeout=180,
        request_timeout=300,
    ),
)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    executor_main(
        steps=[inference_step],
        description="Smoke test: llama.cpp CPU inference (2 workers, 8 documents).",
    )
