import csv

input_test = open("test_output.csv")
test_input = csv.reader(input_test, delimiter=',')
#next(test_input)  # skip header row

output_test = open("test_output_fixed.csv", "wb")
test_output = csv.writer(output_test)

test_output.writerow(["id", "category"])

for r in test_input:
    test_output.writerow(r)