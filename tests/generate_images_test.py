import os
from app.services.Image_generator import process_scraper_results


def main():
    # make sure to run the scraper first to generate the input file before running this test
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    json_path = os.path.join(repo_root, "app", "files", "scraper_results", "scraper_results.txt")

    if not os.path.exists(json_path):
        print(f"Input file not found: {json_path}")
        return

    out_count = process_scraper_results(json_path)
    print(f"Processed {out_count} items and updated file: {json_path}")


if __name__ == "__main__":
    main()
