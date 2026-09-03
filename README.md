# FedMed Flower Server Runtime

## Module Overview

This module contains the **Flower ServerApp integration layer** for FedMed.

The purpose of this module is to connect the framework-independent FedMed
federated-learning core with the **Flower 1.34.0 ServerApp runtime**.

The architecture currently follows:

    Flower ServerApp
            |
            v
    FedMedFlowerStrategy
            |
            v
    FedMed FederatedStrategy
            |
            v
    Aggregator
            |
            v
    FedAvgAggregator

The Flower-specific code is kept in:

    app/server.py

The framework-independent federated-learning logic remains under:

    src/

The application entry point is:

    app/main.py


## What `app/server.py` Contains

`app/server.py` implements the Flower-side server adapter.

### 1. `FedMedFlowerStrategy`

`FedMedFlowerStrategy` extends Flower's:

    flwr.serverapp.strategy.FedAvg

It adapts Flower's ServerApp strategy lifecycle to the FedMed
strategy/aggregation architecture.

The class currently handles:

- Flower node selection
- training message construction
- evaluation message construction
- Flower `Message` handling
- conversion of Flower training replies into FedMed results
- conversion of Flower evaluation replies into FedMed results
- delegation of training aggregation to the FedMed strategy
- delegation of evaluation aggregation to the FedMed strategy


## Flower 1.34.0 Compatibility

The implementation was verified against:

    Flower 1.34.0

Flower 1.34.0 uses the following default ServerApp strategy record keys:

    arrayrecord_key  = "arrays"
    configrecord_key = "config"

Therefore training and evaluation input messages use:

    "arrays"
    "config"

rather than custom record names.

Flower internally constructs training messages using:

    RecordDict({
        self.arrayrecord_key: arrays,
        self.configrecord_key: config,
    })

and sends them with:

    MessageType.TRAIN

Evaluation messages use the same `arrays` and `config` record keys
with:

    MessageType.EVALUATE


## Training Flow

The current training flow is:

    ServerApp
       |
       v
    configure_train()
       |
       +--> select available Flower nodes
       |
       +--> create ConfigRecord
       |
       +--> add server round
       |
       +--> create RecordDict
       |
       +--> construct TRAIN Messages
       |
       v
    Flower SuperNode / ClientApp
       |
       v
    Client training
       |
       v
    FitRes
       |
       v
    Flower RecordDict
       |
       v
    aggregate_train()
       |
       v
    FedMed FederatedStrategy
       |
       v
    FedMed Aggregator


### Training Message

The training message contains:

    fitins.parameters
    fitins.config

The current implementation constructs the corresponding
Flower-compatible RecordDict using the strategy's configured
Flower record keys.


### Training Aggregation

Training aggregation is intentionally delegated to the FedMed
strategy.

The Flower adapter does not implement FedAvg mathematics itself.

The current responsibility split is:

    FedMedFlowerStrategy
        = Flower/runtime policy and adaptation

    FedAvgStrategy
        = FedMed federation strategy

    FedAvgAggregator
        = mathematical parameter aggregation


This preserves the previously established FedMed architecture.


## Evaluation Flow

The evaluation flow is:

    ServerApp
       |
       v
    configure_evaluate()
       |
       +--> select available Flower nodes
       |
       +--> create evaluation ConfigRecord
       |
       +--> add server round
       |
       +--> construct EVALUATE Messages
       |
       v
    Flower SuperNode / ClientApp
       |
       v
    Client evaluation
       |
       v
    EvaluateRes
       |
       v
    Flower RecordDict
       |
       v
    aggregate_evaluate()
       |
       v
    FedMed FederatedStrategy
       |
       v
    aggregated evaluation metrics


## Flower Compatibility Conversion

Flower's compatibility layer represents a training result using:

    fitres.parameters
    fitres.num_examples
    fitres.metrics
    fitres.status

The FedMed Flower adapter converts these records into:

    FederatedFitResult

The conversion validates:

- parameter payload type
- number-of-examples payload
- metric payload type
- numeric metric values

Parameter arrays are copied at the Flower/FedMed boundary to avoid
accidental mutation between the two layers.


Evaluation results are similarly converted into:

    FederatedEvaluateResult


## `app/main.py`

The application entry point is intentionally small.

It constructs the FedMed orchestrator and obtains the Flower
applications from it:

    from src.fl.orchestrator import FedMedOrchestrator

    orchestrator = FedMedOrchestrator()
    client_app, server_app = orchestrator.build_apps()

The module exports:

    client_app
    server_app

Flower therefore loads the ClientApp and ServerApp through the
application configuration in `pyproject.toml`.


## How to Run

### 1. Activate the virtual environment

    cd ~/fedmed
    source .venv/bin/activate


### 2. Start the Flower SuperLink

In Terminal 1:

    cd ~/fedmed
    source .venv/bin/activate
    flower-superlink --insecure

The current local SuperLink starts the following APIs:

    Control API : 9093
    Runtime API : 9091
    Fleet API   : 9092


### 3. Start SuperNode 1

In Terminal 2:

    cd ~/fedmed
    source .venv/bin/activate
    flower-supernode --insecure --superlink 127.0.0.1:9092 --clientappio-api-address 0.0.0.0:9094


### 4. Start SuperNode 2

In Terminal 3:

    cd ~/fedmed
    source .venv/bin/activate
    flower-supernode --insecure --superlink 127.0.0.1:9092 --clientappio-api-address 0.0.0.0:9095


### 5. Run the Flower application

In Terminal 4:

    cd ~/fedmed
    source .venv/bin/activate

    FLWR_LOG_LEVEL=DEBUG flwr run . local-deployment --stream


## Expected Runtime

A successful run currently shows:

    [FedMed] assembling Flower application
    [FedMed] Flower ClientApp assembled
    [FedMed] Flower ServerApp assembled

followed by:

    [FedMed] initial global parameters created
    [FedMed] strategy assembled: FedAvgStrategy -> FedAvgAggregator
    [FedMed] Flower strategy adapter created: FedMedFlowerStrategy

The server then starts the configured number of federated rounds.


For a one-round local deployment, the current successful execution
shows:

    [ROUND 1/1]

followed by:

    configure_train
    training on two selected nodes
    training aggregation
    configure_evaluate
    evaluation on two selected nodes
    evaluation aggregation


## Current Successful Run

The current four-terminal local deployment has been successfully
executed with two SuperNodes.

The latest successful run completed:

    Round: 1/1

Training:

    nodes selected: 2
    batches_processed: 2
    epochs_completed: 1
    num_examples: 16
    train_loss: approximately 0.7721

Evaluation:

    nodes selected: 2
    num_examples: 16
    loss: approximately 0.7087
    accuracy: 0.625

The Flower strategy completed successfully and returned final results.


## Validation

The current implementation has been validated with:

    python -m py_compile app/server.py

    python -c "from app.server import FedMedFlowerStrategy; print('server import OK')"

The server-specific test suite passes:

    31 passed

The complete project test suite currently passes:

    577 passed, 2 warnings

The warnings are existing third-party deprecation warnings from
the installed Typer/Click environment.


## Important Implementation Decision

The Flower adapter does NOT replace the FedMed strategy architecture.

The responsibility boundary is:

    Flower
      |
      | runtime / transport
      v
    FedMedFlowerStrategy
      |
      | federation policy / delegation
      v
    FedMed FederatedStrategy
      |
      | mathematical aggregation
      v
    Aggregator


The `Aggregator` remains responsible for mathematical parameter
aggregation.

The `FederatedStrategy` remains responsible for federated strategy
behavior and delegates mathematical aggregation to the Aggregator.

`FedMedFlowerStrategy` exists to adapt this architecture to Flower's
ServerApp runtime.


## Current Status

### Completed

- Flower 1.34.0 ServerApp integration
- `FedMedFlowerStrategy`
- Flower-compatible `arrays` / `config` record handling
- Flower-compatible training message construction
- Flower-compatible evaluation message construction
- Flower training-result conversion
- Flower evaluation-result conversion
- Strategy-to-Aggregator delegation
- Two-SuperNode local deployment
- Four-terminal Flower dry run
- One-round training execution
- One-round evaluation execution
- Server-side strategy completion
- Server-specific tests
- Full project test suite

### Current Verification

    Flower version: 1.34.0

    Server tests:   31 passed
    Full tests:     577 passed

    Local deployment:
        SuperLink : running
        SuperNode : 2 nodes
        ServerApp : successful
        ClientApp : successful
        Round 1   : successful


## Known Non-Blocking Warning

The Flower application currently reports:

    Recommended property "license" missing in [project]

This is a `pyproject.toml` metadata warning and does not prevent the
application from running.

It is separate from the ServerApp runtime implementation.


## Files Relevant to This Module

    app/
    ├── main.py
    ├── server.py
    └── client.py

    src/
    ├── fl/
    │   └── orchestrator.py
    └── ...

    tests/
    └── test_app_server.py

    pyproject.toml


## Daily Development Check

Before committing changes to this module:

    python -m py_compile app/server.py

    python -c "from app.server import FedMedFlowerStrategy; print('server import OK')"

    pytest -q tests/test_app_server.py

    pytest -q

A successful state is:

    31 passed

and:

    577 passed, 2 warnings