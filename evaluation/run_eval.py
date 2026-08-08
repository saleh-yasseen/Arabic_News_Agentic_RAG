import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing_eval import run_routing_eval, print_routing_report
from retrieval_eval import run_retrieval_eval, print_retrieval_report
from generation_eval import run_generation_eval, print_generation_report
from latency_eval import run_latency_eval, print_latency_report
from _logging import save_evaluation_run, write_summary, current_run_folder


def run_all():
    sections = {}        # eval_type -> results, used to build the summary
    detail_sections = {} # eval_type -> query_details, used for the combined log

    print("=" * 60)
    routing, routing_details = run_routing_eval()
    print_routing_report(routing)
    save_evaluation_run("routing", routing, routing_details)
    sections["routing"] = routing
    detail_sections["routing"] = routing_details

    print("\n" + "=" * 60)
    retrieval, retrieval_details = run_retrieval_eval()
    print_retrieval_report(retrieval)
    if retrieval is not None:
        save_evaluation_run("retrieval", retrieval, retrieval_details)
    sections["retrieval"] = retrieval
    detail_sections["retrieval"] = retrieval_details

    print("\n" + "=" * 60)
    generation, generation_details = run_generation_eval()
    print_generation_report(generation)
    save_evaluation_run("generation", generation, generation_details)
    sections["generation"] = generation
    detail_sections["generation"] = generation_details

    print("\n" + "=" * 60)
    latency, latency_details = run_latency_eval()
    print_latency_report(latency)
    save_evaluation_run("latency", latency, latency_details)
    sections["latency"] = latency
    detail_sections["latency"] = latency_details

    # Combined "all" payload — same folder as the per-section files.
    combined = {"sections": sections}
    combined_details = [
        {"section": name, "details": details}
        for name, details in detail_sections.items()
    ]
    save_evaluation_run("all", combined, combined_details)

    # Single summary aggregating all four sections.
    _, folder = current_run_folder()
    summary_paths = write_summary(folder, sections)

    print("\n" + "=" * 60)
    print("Summary written to:")
    print(f"  {summary_paths['summary_json']}")
    print(f"  {summary_paths['summary_md']}")

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
