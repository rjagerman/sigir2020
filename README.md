# Accelerated Convergence for Counterfactual Learning to Rank

Source code with our SIGIR2020 paper.
To run all experiments, build the plots and create the tables that appear in the paper run the following:

    export YAHOO_DIR=/path/to/yahoo/set1/
    export ISTELLA_DIR=/path/to/istella/sample/
    make -j$(nproc)

Or, if you are on macOS:

    export YAHOO_DIR=/path/to/yahoo/set1/
    export ISTELLA_DIR=/path/to/istella/sample/
    make -j$(sysctl -n hw.logicalcpu)

This can take significant amount of time depending on the number of CPU cores available.
