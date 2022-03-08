# Sparse Compiler and Design Implementation (SCADI) 

## SCADI Compiler
```
TODO
```

## SCADI Simulator

### Running Tests
The simulator uses pytest to run tests

To run all tests type 
```
cd 
pytest
```

Use the following pytest optional arguments below
```
--debug-sim                 Turn on debug mode for sim
--count <n>                 Repeat each test for n iterations 
-k <testname>[<paramlist>]  Run only tests with testname and paramlist
-vv                         Double verbose
-s                          Forward printouts to stdout
--full-trace                Print full trace to stdout
```


### Test Naming Convention
Full kernel tests follow the naming convention `test_<kernel>_<rand|direct>_<outformat>_<in1format>_<in2format>_...<innformat>` where: 
1. `<kernel>` is the name of the tensor algebra kernel being tested (e.g. mat_elemmul, mat_mul, vec_elemmul, etc.)
2. `*format` takes on `u | c | s` for formats uncompressed, compressed, singleton respectively
3. `<rand|direct> specifies if the test is _randomly generated_ or a _directed (handwritten)_ test

Primitive unit tests follow the naming convention `test_<primitive>_<feature>_<order>` where:
1. `<primitive>` is the name of the primitive being tested (e.g. array, intersect, union, etc.)
2. `<feature>` is the name of the feature being tested (e.g. for an array we can test both loads and stores)
3. `<order>` is the name of the order of stream being tested (1d for vectors,
2d for matrices, ..., and nd for all dimensions/tensor orders, etc.)
 

### Directory Structure
```
sim
│   
│       
│
└───src
│   │   base.py
│   │   joiner.py
│   │   ...         # All primitive block classes
│   
└───test
│   │   test.py
│   │   file022.txt
│   │
│   └───apps
│       │   test_mat_elemmul.py
│       │   ...     # Full kernel/expression tests
│
└───────primitives
        │   test_joiner.py
        │   ...     # Primitive unit tests
   
```
