"""Terminal version of RaceMind AI.  Usage:  python rag_pipeline.py"""

import rag


def main():
    question = input("\nAsk RaceMind: ").strip()
    result = rag.ask(question)

    print("\n" + "=" * 80)
    print("RACEMIND ANSWER")
    print("=" * 80)
    print(result["answer"])

    print("\n" + "=" * 80)
    print("SOURCES")
    print("=" * 80)
    for i, src in enumerate(result["sources"], start=1):
        print(f"{i}. Page {src['page']} - {src['source']}")


if __name__ == "__main__":
    main()
