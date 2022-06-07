BENCHES := "" 

# Set OPENMP=ON if compiling TACO with OpenMP support.
ifeq ($(OPENMP),)
OPENMP := "OFF"
endif
# Set NEVA=ON if compiling on the Stanford cluster (Neva or Kiwi).
ifeq ($(NEVA),)
NEVA := "OFF"
endif
# Set GEN=1 if you would like to generate "other" tensors for performance into a file
ifeq ($(GEN),)
GEN := "0"
endif

benches_name := $(patsubst %.py,%,$(BENCHES))
benches_name := $(subst /,_,$(benches_name))
benches_name := $(subst *,_,$(benches_name))
# Taco Specific Flags
ifeq ($(TACO_OUT),)
TACO_OUT := results-cpu/$(benches_name)_benches_$(shell date +%Y_%m_%d_%H%M%S).csv
endif

ifeq ("$(NEVA)","ON")
CMD := OMP_PROC_BIND=true LD_LIBRARY_PATH=compiler/build/lib/:$(LD_LIBRARY_PATH) numactl -C 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 -m 0 compiler/build/taco-bench $(BENCHFLAGS)
else
CMD := LD_LIBRARY_PATH=compiler/build/lib/:$(LD_LIBRARY_PATH) compiler/build/taco-bench $(BENCHFLAGS)
endif

export SUITESPARSE_PATH = /nobackup/owhsu/sparse-datasets/suitesparse/
export FROSTT_PATH = /nobackup/owhsu/sparse-datasets/frostt/
export SUITESPARSE_FORMATTED_PATH=/nobackup/owhsu/sparse-datasets/suitesparse-formatted
export TACO_TENSOR_PATH=/nobackup/owhsu/sparse-datasets/

# ---- Run SAM python simulator stuff ----
csv: 
	scripts/pytest_suitesparse_with_benchmarks.sh

run: submodules
	./scripts/pytest_suitesparse.sh

tests: sam 
	python scripts/test_generating_code.py
	make run

.PHONY: formats
formats:
	rm -rf ${SUITESPARSE_FORMATTED_PATH}/*
	set -e && ./scripts/generate_suitesparse_formats.sh

# ---- Build taco and make sam graphs ----
sam: taco/build 
	 cd compiler && bash -xe ./sam-kernels.sh

taco/build: submodules
	 mkdir -p compiler/taco/build 
	 cd compiler/taco/build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j8

# ---- Build and run taco-bench (cpu baseline) ----
taco-bench: compiler/build/taco-bench
	export GEN=$(GEN)
	mkdir -p $(TACO_TENSOR_PATH)/other
	mkdir -p results-cpu 
ifeq ($(BENCHES),"")
	$(CMD) --benchmark_out_format="csv" --benchmark_out="$(TACO_OUT)" --benchmark_repetitions=10 --benchmark_counters_tabular=true
else
	$(CMD) --benchmark_filter="$(BENCHES)" --benchmark_out_format="csv" --benchmark_out="$(TACO_OUT)" --benchmark_repetitions=10 --benchmark_counters_tabular=true
endif

compiler/build/taco-bench: submodules compiler/benchmark/googletest
	mkdir -p compiler/build/ && cd compiler/build/ && cmake -DOPENMP=$(OPENMP) -DNEVA=$(NEVA) ../ && $(MAKE) taco-bench

compiler/benchmark/googletest: submodules
	if [ ! -d "compiler/benchmark/googletest" ] ; then git clone https://github.com/google/googletest compiler/benchmark/googletest; fi

# ---- Validate taco-bench and SAM python simulator
validate-bench: compiler/build/taco-bench validation-path
ifeq ($(BENCHES),"")
	$(CMD) --benchmark_repetitions=1
else
	$(CMD) --benchmark_filter="$(BENCHES)" --benchmark_repetitions=1
endif

.PHONY: validation-path
validation-path:
ifeq ($(VALIDATION_OUTPUT_PATH),)
	$(error VALIDATION_OUTPUT_PATH is undefined)
endif

# Separate target to run the SAM sim python benchmarks with taco cross validation logic.
# validate-sam-bench: validation-path
# 	pytest $(IGNORE_FLAGS) $(BENCHFLAGS) $(BENCHES)

# ---- Setup proper environment stuff ----
.PHONY: env
env:
	export SUITESPARSE_PATH=/nobackup/owhsu/sparse-datasets/suitesparse/
	export FROSTT_PATH=/nobackup/owhsu/sparse-datasets/frostt/
	export SUITESPARSE_FORMATTED_PATH=/nobackup/owhsu/sparse-datasets/suitesparse-formatted

.PHONY: pydepends
pydepends:
	conda env export > environment.yml
	pip list --format=freeze > requirements.txt


.PHONY: submodules
submodules:
	@if git submodule status | egrep -q '^[-]|^[+]' ; then \
		echo "INFO: Need to reinitialize git submodules"; \
		git submodule update --init; \
	fi

.PHONY: clean 
clean:
	cd compiler && rm -rf sam-outputs
	cd compiler/taco/build && make clean
