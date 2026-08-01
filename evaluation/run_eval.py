import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing_eval import run_routing_eval, print_routing_report
from retrieval_eval import run_retrieval_eval, print_retrieval_report
from generation_eval import run_generation_eval, print_generation_report
from latency_eval import run_latency_eval, print_latency_report


def run_all():
    print("=" * 60)
    routing = run_routing_eval()
    print_routing_report(routing)

    print("\n" + "=" * 60)
    retrieval = run_retrieval_eval()
    print_retrieval_report(retrieval)

    print("\n" + "=" * 60)
    generation = run_generation_eval()
    print_generation_report(generation)

    print("\n" + "=" * 60)
    latency = run_latency_eval()
    print_latency_report(latency)

    print("\n" + "=" * 60)
    print("README summary block\n")
    print("## Evaluation\n")
    print(f"Router Accuracy\n{routing['overall_accuracy']*100:.1f}%\n")

    if retrieval:
        best_mode = max(retrieval["modes"].items(), key=lambda x: x[1]["recall_at_5"])
        print(f"Retrieval\nRecall@5 ({best_mode[0]})\n{best_mode[1]['recall_at_5']*100:.0f}%\n")

    gen_avg = generation["averages"]
    if gen_avg["groundedness"] is not None:
        print(f"Generation\nGroundedness\n{gen_avg['groundedness']}/5\n")

    print(f"Latency\n{latency['averages']['total_ms']/1000:.2f} sec")


if __name__ == "__main__":
    run_all()
