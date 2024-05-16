import sys
import csv
import json
import pathlib


def csv2json(source_file, output_file):
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
    if len(sys.argv) != 2:
        print("Usage: python csv2json.py <source_file>")
        sys.exit(1)
    input = pathlib.Path(sys.argv[1])
    if not input.exists():
        print(f"File {input} does not exist.")
        sys.exit(1)
    output = input.with_suffix(".json")

    csv2json(input, output)
