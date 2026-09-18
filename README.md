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

## Federated Learning Experiments

The following experiments have been completed using the Flower 1.34.0
multi-node runtime. The experiments are intended to validate FedMed's
federated-learning behavior under different runtime conditions.

### Experiment Discipline

Each experiment follows a controlled approach:

1. Define the hypothesis or experimental objective.
2. Keep unrelated configuration variables fixed.
3. Change only the variable under investigation.
4. Run the federated training workload.
5. Record training/evaluation metrics and runtime evidence.
6. Compare the results and document the observations.

---

### E1 — Baseline Reproducibility

**Objective:** Verify that the same FedMed federated-learning configuration
produces reproducible results across repeated runs.

Configuration:

    Clients: 2
    Rounds: 3
    Train fraction: 1.0
    Evaluation fraction: 1.0
    Local epochs: 1
    Partition: current IID setup
    Strategy: FedAvgStrategy
    Aggregator: FedAvgAggregator

The experiment was executed twice using the same configuration.

Round parameter fingerprints:

    Round 1: 5d2399307f878547 -> 166dbbaac8c674b6
    Round 2: 166dbbaac8c674b6 -> 2c9f2041b7a13ade
    Round 3: 2c9f2041b7a13ade -> 32346c94ed9fdb6f

Aggregated metrics:

    Train loss:
        Round 1: 0.8096474260
        Round 2: 0.8063939661
        Round 3: 0.8031985164

    Evaluation loss:
        Round 1: 0.6489310861
        Round 2: 0.6493559479
        Round 3: 0.6498010904

    Accuracy:
        50.00% in every round

    Examples:
        16 per round

**Result:**

The repeated runs produced the same parameter fingerprints and metrics.
This validates deterministic/reproducible behavior for the current baseline
configuration.

---

### E2 — Client Count

**Objective:** Observe federated-learning behavior when the number of
participating Flower clients changes.

The experiments used 3 federated rounds with 100% training and evaluation
participation.

#### E2-A — 1 Client

    Training clients per round: 1
    Evaluation clients per round: 1
    Examples per round: 8

Metrics:

    Train loss:
        Round 1: 0.8189847767
        Round 2: 0.8152351081
        Round 3: 0.8115414977

    Evaluation loss:
        Round 1: 0.6616895199
        Round 2: 0.6618472934
        Round 3: 0.6620339751

    Accuracy:
        50.00% in every round

#### E2-B — 2 Clients

    Training clients per round: 2
    Evaluation clients per round: 2
    Examples per round: 16

Metrics:

    Train loss:
        Round 1: 0.8096474260
        Round 2: 0.8063939661
        Round 3: 0.8031985164

    Evaluation loss:
        Round 1: 0.6489310861
        Round 2: 0.6493559479
        Round 3: 0.6498010904

    Accuracy:
        50.00% in every round

#### E2-C — 3 Clients

    Training clients per round: 3
    Evaluation clients per round: 3
    Examples per round: 24

Metrics:

    Train loss:
        Round 1: 0.7625652552
        Round 2: 0.7611813347
        Round 3: 0.7598178188

    Evaluation loss:
        Round 1: 0.6903412938
        Round 2: 0.6900853515
        Round 3: 0.6898385584

    Accuracy:
        41.67% in every round

**Observation:**

Increasing the number of clients did not automatically improve evaluation
accuracy in the current experiment. The 3-client experiment achieved lower
training loss but lower evaluation accuracy than the 1- and 2-client runs.

The experiment also changes the total amount of data because each client
currently contributes 8 examples. Therefore, this experiment measures client
count together with the corresponding increase in total participating data;
it does not isolate client count as a completely independent variable.

---

### E3 — Training Client Participation Fraction

**Objective:** Validate Flower training-client fraction selection and observe
the effect of partial client participation while keeping evaluation
participation at 100%.

The experiments used:

    Available clients: 4
    Rounds: 3
    Evaluation fraction: 1.0
    Local epochs: 1

The runtime selection logic uses the configured participation fraction to
select the required number of available Flower nodes.

#### E3-A — 100% Training Participation

    Training fraction: 1.00
    Training clients: 4
    Evaluation clients: 4
    Examples per training round: 32

Metrics:

    Train loss:
        Round 1: 0.7766701356
        Round 2: 0.7748080865
        Round 3: 0.7729736418

    Evaluation loss:
        Round 1: 0.6831279024
        Round 2: 0.6829234138
        Round 3: 0.6827315167

    Accuracy:
        43.75% in every round

#### E3-B — 75% Training Participation

    Training fraction: 0.75
    Training clients: 3
    Evaluation clients: 4
    Examples per training round: 24

Metrics:

    Train loss:
        Round 1: 0.7625652552
        Round 2: 0.7611813347
        Round 3: 0.7598178188

    Evaluation loss:
        Round 1: 0.6831628382
        Round 2: 0.6829900295
        Round 3: 0.6828266159

    Accuracy:
        43.75% in every round

#### E3-C — 50% Training Participation

    Training fraction: 0.50
    Training clients: 2
    Evaluation clients: 4
    Examples per training round: 16

Metrics:

    Train loss:
        Round 1: 0.6990955323
        Round 2: 0.6990120113
        Round 3: 0.6989294589

    Evaluation loss:
        Round 1: 0.6494003683
        Round 2: 0.6494136974
        Round 3: 0.6494279876

    Accuracy:
        Round 1: 56.25%
        Round 2: 56.25%
        Round 3: 59.375%

#### E3-D — 25% Training Participation

    Training fraction: 0.25
    Training clients: 1
    Evaluation clients: 4
    Examples per training round: 8

Metrics:

    Train loss:
        Round 1: 0.7484648526
        Round 2: 0.7473103702
        Round 3: 0.7461675704

    Evaluation loss:
        Round 1: 0.6494098157
        Round 2: 0.6494439542
        Round 3: 0.6494901925

    Accuracy:
        Round 1: 56.25%
        Round 2: 59.375%
        Round 3: 59.375%

**Runtime validation:**

    100% -> 4 training clients
     75% -> 3 training clients
     50% -> 2 training clients
     25% -> 1 training client

Evaluation remained at 4 clients for all E3 experiments.

Parameter fingerprints changed across every federated round, confirming that
the global model parameters continued to evolve during training.

**Observation:**

The participation-fraction mechanism works correctly in the real Flower
multi-node runtime. Lower participation did not cause runtime or aggregation
failures in these experiments.

The 50% and 25% experiments produced higher evaluation accuracy than the
100% and 75% experiments in these particular runs. This should not be
interpreted as evidence that lower participation is inherently better.
Client selection is deterministic in the current implementation, and
different participation fractions result in different sets of training
clients. More controlled repetitions would therefore be required to
attribute performance differences specifically to the participation
fraction.

Runtime measurements were also recorded, but the runs experienced Flower
logstream reconnections and startup overhead. Therefore, runtime differences
should not be treated as a clean measurement of participation-efficiency
scaling.

---
Experiment Status

Completed:

    E1 — Baseline Reproducibility               [COMPLETED]
    E2 — Client Count                           [COMPLETED]
    E3 — Training Client Participation Fraction [COMPLETED]
    E4 — Number of Federated Rounds             [COMPLETED]
    E5 — IID vs Non-IID Data                    [COMPLETED]
    E6 — Client Failure / Dropout               [COMPLETED]

### E6 — Client Failure / Dropout Results

**Configuration**

    Flower SuperNodes: 4
    Partitions: 0, 1, 2, 3
    Training participation: 100%
    Evaluation participation: 100%
    Federated rounds: 3
    Data partitioning: IID
    Controlled failure: partition 0 during Round 2
    Failure scope: TRAIN only

**Observed results**

| Round | Successful training clients | Training examples | Train loss | Eval loss | Accuracy |
|---|---:|---:|---:|---:|---:|
| 1 | 4/4 | 32 | 0.687063 | 0.686760 | 0.5000 |
| 2 | 3/4 | 24 | 0.701005 | 0.686616 | 0.5000 |
| 3 | 4/4 | 32 | 0.686734 | 0.686436 | 0.5000 |

**Parameter fingerprints**

| Round | Input | Output |
|---|---|---|
| 1 | `5d2399307f878547` | `25b09817ca2394e1` |
| 2 | `25b09817ca2394e1` | `fdfe7049db24c893` |
| 3 | `fdfe7049db24c893` | `9ad410e9f5e5b295` |

**Failure handling evidence**

During Round 2, partition 0 raised the controlled E6 failure:

    E6 controlled client dropout: partition=0, round=2

The Flower server received the failed training reply and ignored it:

    ignoring failed training reply from node 5740701070240325215

Aggregation then continued:

    Delegating Flower training aggregation to FedMed Strategy for round 2.
    ROUND 2 OUTPUT fingerprint=fdfe7049db24c893

Round 2 therefore completed using the three successful training clients. Evaluation still used all four clients, and Round 3 returned to 4/4 successful training clients.

**Conclusion**

E6 successfully demonstrates controlled client-failure tolerance in the Flower runtime. A training client can fail during a federated round without aborting the round: the failed reply is ignored, the remaining successful client updates are aggregated, evaluation continues normally, and subsequent rounds proceed successfully.


Planned:

    E9 — Centralized vs Federated Training

### E7 — Local Epochs

E7 evaluates the effect of local training epochs on the federated learning
workflow while keeping the federated runtime configuration fixed.

**Configurations**

    E7-A → local_epochs = 1
    E7-B → local_epochs = 2
    E7-C → local_epochs = 5

**Fixed configuration**

    Flower SuperNodes: 4
    Partitions: 0, 1, 2, 3
    Training participation: 100%
    Evaluation participation: 100%
    Federated rounds: 3
    Data partitioning: IID
    Batch size: 4
    Learning rate: 0.01
    Optimizer: SGD
    Seed: 42
    Controlled failure: disabled

#### E7-A — Local Epochs = 1

| Round | Epochs | Train Loss | Eval Loss | Accuracy | Train Examples |
|---|---:|---:|---:|---:|---:|
| 1 | 1 | 0.687063 | 0.686760 | 0.5000 | 32 |
| 2 | 1 | 0.686876 | 0.686577 | 0.5000 | 32 |
| 3 | 1 | 0.686694 | 0.686398 | 0.5000 | 32 |

**Parameter fingerprints**

| Round | Input | Output |
|---|---|---|
| 1 | `5d2399307f878547` | `25b09817ca2394e1` |
| 2 | `25b09817ca2394e1` | `8d8daa3109b91d4d` |
| 3 | `8d8daa3109b91d4d` | `e668691dd739dc09` |

Runtime: **250.32s**

Client-side metrics confirmed `epochs_completed = 1`,
`batches_processed = 2`, and `num_examples = 32` per round.

#### E7-B — Local Epochs = 2

| Round | Epochs | Train Loss | Eval Loss | Accuracy | Train Examples |
|---|---:|---:|---:|---:|---:|
| 1 | 2 | 0.683617 | 0.686575 | 0.5000 | 64 |
| 2 | 2 | 0.683248 | 0.686220 | 0.5000 | 64 |
| 3 | 2 | 0.682896 | 0.685882 | 0.5000 | 64 |

**Parameter fingerprints**

| Round | Input | Output |
|---|---|---|
| 1 | `5d2399307f878547` | `a2a88244ee48d7e9` |
| 2 | `a2a88244ee48d7e9` | `6434ec3385c1ca86` |
| 3 | `6434ec3385c1ca86` | `ca732d9ce8ca8362` |

Runtime: **250.24s**

Client-side metrics confirmed `epochs_completed = 2`,
`batches_processed = 4`, and `num_examples = 64` per round.

#### E7-C — Local Epochs = 5

| Round | Epochs | Train Loss | Eval Loss | Accuracy | Train Examples |
|---|---:|---:|---:|---:|---:|
| 1 | 5 | 0.673805 | 0.686037 | 0.5000 | 160 |
| 2 | 5 | 0.672913 | 0.685231 | 0.5000 | 160 |
| 3 | 5 | 0.672117 | 0.684518 | 0.5000 | 160 |

**Parameter fingerprints**

| Round | Input | Output |
|---|---|---|
| 1 | `5d2399307f878547` | `1b889617a072c9a9` |
| 2 | `1b889617a072c9a9` | `0bb514318ba100d0` |
| 3 | `0bb514318ba100d0` | `c08dc578e6a33acf` |

Runtime: **311.00s**

Client-side metrics confirmed `epochs_completed = 5`,
`batches_processed = 10`, and `num_examples = 160` per round.

#### E7 Summary

| Metric | E7-A: 1 Epoch | E7-B: 2 Epochs | E7-C: 5 Epochs |
|---|---:|---:|---:|
| Final train loss | 0.686694 | 0.682896 | 0.672117 |
| Final eval loss | 0.686398 | 0.685882 | 0.684518 |
| Accuracy | 0.5000 | 0.5000 | 0.5000 |
| Train examples/round | 32 | 64 | 160 |
| Batches/client | 2 | 4 | 10 |
| Runtime | 250.32s | 250.24s | 311.00s |

Across these runs, increasing local epochs was associated with lower training
loss and slightly lower evaluation loss, while measured accuracy remained
0.5000. The 5-epoch run required more measured execution time. These are
observations from this experimental configuration and are not treated as
general conclusions about local epoch selection.

All three E7 configurations completed all three federated rounds with 4/4
training clients and 4/4 evaluation clients participating in every round.
Parameter fingerprints advanced across every round, confirming continued
global model updates.

### E8 — Data Quantity Imbalance

E8 evaluates federated training with unequal quantities of local training data
while keeping evaluation data balanced. The purpose is to exercise the
existing example-weighted aggregation path under heterogeneous client data
sizes.

**Configuration**

    Flower SuperNodes: 4
    Partitions: 0, 1, 2, 3
    Training participation: 100%
    Evaluation participation: 100%
    Federated rounds: 3
    Data partitioning: IID
    Local epochs: 5
    Batch size: 4
    Learning rate: 0.01
    Optimizer: SGD
    Seed: 42
    Controlled failure: disabled

**Training-data distribution**

    client_0 → 4 examples
    client_1 → 8 examples
    client_2 → 8 examples
    client_3 → 12 examples

    Total → 32 examples

Evaluation data remained balanced:

    client_0 → 8 examples
    client_1 → 8 examples
    client_2 → 8 examples
    client_3 → 8 examples

    Total → 32 examples

#### E8 Results

| Round | Train Loss | Eval Loss | Accuracy | Eval Examples |
|---|---:|---:|---:|---:|
| 1 | 0.680932 | 0.686110 | 0.5000 | 32 |
| 2 | 0.680199 | 0.685398 | 0.5000 | 32 |
| 3 | 0.679574 | 0.684794 | 0.5000 | 32 |

**Parameter fingerprints**

| Round | Input | Output |
|---|---|---|
| 1 | `5d2399307f878547` | `74f3bd54e6ec7312` |
| 2 | `74f3bd54e6ec7312` | `e0c60566724a0461` |
| 3 | `e0c60566724a0461` | `51c4208f03172894` |

Runtime: **159.91s**

All three rounds completed with 4/4 training clients and 4/4 evaluation
clients participating in every round.

The aggregated client-side training metrics reported `num_examples = 160`
per round, corresponding to 32 training examples processed for 5 local
epochs. The reported `batches_processed = 11.25` is consistent with the
example-weighted aggregation of the heterogeneous client workloads.

Evaluation remained fixed at 32 examples per round. Accuracy remained
0.5000 across all three rounds, while training loss decreased from 0.680932
to 0.679574 and evaluation loss decreased from 0.686110 to 0.684794.

These observations describe this experimental configuration and are not
treated as general conclusions about data imbalance or aggregation behavior.
