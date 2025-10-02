import gradio as gr
import os

SCRIPT_NAME = "p2pfl_cli.py"  # Name of your CLI script

# Define choices
framework_choices = ["xgboost", "pytorch"]
topology_choices = ["LINE", "RING", "STAR", "FULL"]
protocol_choices = ["memory", "grpc"]
partition_choices = ["randomiid", "dirichlet"]
aggregator_choices = [
    "fedxgbcyclic", "fedxgbbagging",
    "fedavg", "fedadam", "fedadagrad", "fedmedian",
    "fedprox", "fedyogi", "scaffold", "krum"
]
dataset_choices = ["", "ton-iot", "ugr-16", "unr-idd", "unsw-nb15", "zeekdata"]

def build_federated_command(framework, protocol, dataset,
                            data_csv, train_csv, test_csv,
                            nodes, rounds, seed, topology,
                            partition_strategy, target_name, agregator):
    script_path = os.path.abspath(SCRIPT_NAME)
    cmd = ["poetry", "run", "python", script_path,
           "--framework", framework,
           "--protocol", protocol]
    if dataset:
        cmd += ["--dataset", dataset]

    if data_csv:
        cmd += ["--data_csv", data_csv]
    else:
        if train_csv:
            cmd += ["--train_csv", train_csv]
        if test_csv:
            cmd += ["--test_csv", test_csv]

    cmd += [
        "--nodes", str(int(nodes)),
        "--rounds", str(int(rounds)),
        "--seed", str(int(seed)),
        "--topology", topology,
        "--partition_strategy", partition_strategy,
        "--target_name", target_name,
        "--agregator", agregator,
    ]

    return " ".join(cmd)

def create_ui():
    with gr.Blocks(css="""
        .section { margin-top: 1.5em; }
        .input-row { margin-bottom: 1em; }
        textarea { font-family: monospace; }
    """) as demo:
        gr.Markdown("# 📦 Federated CLI Command Builder")

        with gr.Row(elem_classes="section"):
            with gr.Column():
                gr.Markdown("### General")
                framework = gr.Dropdown(framework_choices, value="xgboost",
                                        label="Framework")
                protocol  = gr.Dropdown(protocol_choices, value="memory",
                                        label="Communication Protocol")
                dataset   = gr.Dropdown(dataset_choices, value="",
                                        label="Predefined Dataset")

            with gr.Column():
                gr.Markdown("### CSV Inputs")
                data_csv  = gr.Textbox(label="Single CSV Path",
                                       placeholder="(leave blank to use train/test)")
                train_csv = gr.Textbox(label="Train CSV Path")
                test_csv  = gr.Textbox(label="Test CSV Path")

        with gr.Row(elem_classes="section input-row"):
            with gr.Column():
                gr.Markdown("### Federation Settings")
                nodes  = gr.Number(value=3, label="Number of Nodes")
                rounds = gr.Number(value=3, label="Number of Rounds")
                seed   = gr.Number(value=42, label="Random Seed")

            with gr.Column():
                topology            = gr.Dropdown(topology_choices, value="LINE",
                                                 label="Topology Type")
                partition_strategy  = gr.Dropdown(partition_choices, value="dirichlet",
                                                 label="Partition Strategy")
                target_name         = gr.Textbox(value="outcome",
                                                 label="Target Column Name")
                agregator           = gr.Dropdown(aggregator_choices,
                                                 value="fedxgbcyclic",
                                                 label="Aggregator")

        output_box = gr.Textbox(label="Generated Command",
                                lines=2, interactive=False)
        build_btn  = gr.Button("🚀 Build Command")
        copy_btn   = gr.Button("📋 Copy Command")
        error_box  = gr.Textbox(label="Error", lines=2, interactive=False, visible=False)

        def validate_inputs(dataset, data_csv, train_csv, test_csv):
            # Contar cuántos métodos de entrada están presentes
            count = 0
            if dataset:
                count += 1
            if data_csv:
                count += 1
            if train_csv or test_csv:
                # Solo cuenta si ambos están presentes
                if train_csv and test_csv:
                    count += 1
                elif train_csv or test_csv:
                    return False, "Debes proporcionar tanto Train CSV como Test CSV."
            if count != 1:
                return False, "Debes seleccionar solo UNA de las siguientes opciones: un dataset predefinido, un solo archivo CSV, o ambos archivos de train y test."
            return True, ""

        def build_command_with_validation(framework, protocol, dataset,
                                          data_csv, train_csv, test_csv,
                                          nodes, rounds, seed, topology,
                                          partition_strategy, target_name, agregator):
            valid, msg = validate_inputs(dataset, data_csv, train_csv, test_csv)
            if not valid:
                return gr.update(value="", visible=True), msg
            cmd = build_federated_command(framework, protocol, dataset,
                                          data_csv, train_csv, test_csv,
                                          nodes, rounds, seed, topology,
                                          partition_strategy, target_name, agregator)
            return cmd, ""

        build_btn.click(
            build_command_with_validation,
            inputs=[
                framework, protocol, dataset,
                data_csv, train_csv, test_csv,
                nodes, rounds, seed,
                topology, partition_strategy,
                target_name, agregator
            ],
            outputs=[output_box, error_box]
        )
        copy_btn.click(
            lambda x: x,  # Devuelve el texto tal cual
            inputs=output_box,
            outputs=output_box,
            js="navigator.clipboard.writeText(document.querySelector('textarea[aria-label=Generated Command]').value);"
        )

    return demo

if __name__ == "__main__":
    app = create_ui()
    app.launch(share=False)
