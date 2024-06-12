import argparse
import csv
import json
import pathlib


def csv_to_json(source_file, output_file):
    jsonArray = []

    with open(source_file, encoding="utf-8") as csvf:
        # load csv file data using csv reader
        csvReader = csv.reader(csvf)
        for row in csvReader:
            jsonArray.append(row)

    # remove ` and trailing _
    for e in jsonArray[1:]:
        e[0] = e[0].split("`")[1]

    # remove column names
    jsonArray = {"data": jsonArray[1:]}

    with open(output_file, "w", encoding="utf-8") as jsonf:
        jsonString = json.dumps(jsonArray, indent=4)
        jsonf.write(jsonString)
        jsonf.write("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="csv_json_results_converter.py",
        description="Convert CSV results file in JSON for doc generation",
    )
    parser.add_argument("source_file", type=pathlib.Path, help="CSV result file")
    args = parser.parse_args()
    output_file = args.source_file.with_suffix(".json")

    csv_to_json(args.source_file, output_file)
