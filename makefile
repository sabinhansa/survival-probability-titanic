ifdef print
    OUTPUT := /dev/stdout
else
    OUTPUT := /dev/null
endif

all: run_all

part1:
	@python3 PART1.py > $(OUTPUT) 2>&1

part2:
	@python3 PART2.py > $(OUTPUT) 2>&1

part2_test:
	@python3 PART2_TEST.py > $(OUTPUT) 2>&1

separation:
	@python3 SEPARATION.py > $(OUTPUT) 2>&1

training:
	@python3 TRAINING.py > $(OUTPUT) 2>&1

testing:
	@python3 TESTING.py > $(OUTPUT) 2>&1

probabilities:
	@python3 PROBABILITIES.py > $(OUTPUT) 2>&1

run_all: part1 part2 part2_test training testing probabilities

.PHONY: all part1 part2 part2_test training testing testing probabilities run_all

