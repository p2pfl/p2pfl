import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import subprocess
import sys
import os
import threading

SCRIPT_NAME = "p2pfl_cli.py"

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


class FedCliApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Federated CLI Runner")
        self._make_widgets()
        self._layout_widgets()

    def _make_widgets(self):
        # Dropdowns
        self.framework = ttk.Combobox(self, values=framework_choices, state="readonly")
        self.framework.set(framework_choices[0])
        self.protocol = ttk.Combobox(self, values=protocol_choices, state="readonly")
        self.protocol.set(protocol_choices[0])
        self.topology = ttk.Combobox(self, values=topology_choices, state="readonly")
        self.topology.set(topology_choices[0])
        self.partition = ttk.Combobox(self, values=partition_choices, state="readonly")
        self.partition.set(partition_choices[1])
        self.aggregator = ttk.Combobox(self, values=aggregator_choices, state="readonly")
        self.aggregator.set(aggregator_choices[0])
        self.dataset = ttk.Combobox(self, values=dataset_choices, state="readonly")
        self.dataset.set(dataset_choices[0])

        # Numeric entries
        self.nodes = tk.Spinbox(self, from_=1, to=100, width=5)
        self.rounds = tk.Spinbox(self, from_=1, to=1000, width=5)
        self.seed = tk.Spinbox(self, from_=0, to=999999, width=7)
        self.nodes.delete(0, "end"); self.nodes.insert(0, "3")
        self.rounds.delete(0, "end"); self.rounds.insert(0, "3")
        self.seed.delete(0, "end"); self.seed.insert(0, "42")

        # Text entries
        self.data_csv = tk.Entry(self, width=40)
        self.train_csv = tk.Entry(self, width=20)
        self.test_csv = tk.Entry(self, width=20)
        self.target_name = tk.Entry(self, width=20)
        self.target_name.insert(0, "outcome")

        # Buttons to browse
        self.btn_data = ttk.Button(self, text="…", width=2,
                                   command=lambda: self._browse_file(self.data_csv))
        self.btn_train = ttk.Button(self, text="…", width=2,
                                    command=lambda: self._browse_file(self.train_csv))
        self.btn_test = ttk.Button(self, text="…", width=2,
                                   command=lambda: self._browse_file(self.test_csv))

        # Run button & output
        self.btn_run = ttk.Button(self, text="Run", command=self._run_cli)
        self.output = scrolledtext.ScrolledText(self, width=80, height=20, wrap="word")

    def _layout_widgets(self):
        pad = {"padx": 5, "pady": 3}
        row = 0

        ttk.Label(self, text="Framework:").grid(row=row, column=0, sticky="e", **pad)
        self.framework.grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(self, text="Protocol:").grid(row=row, column=2, sticky="e", **pad)
        self.protocol.grid(row=row, column=3, sticky="w", **pad)

        row += 1
        ttk.Label(self, text="Single CSV:").grid(row=row, column=0, sticky="e", **pad)
        self.data_csv.grid(row=row, column=1, sticky="w", **pad)
        self.btn_data.grid(row=row, column=2, sticky="w", **pad)

        row += 1
        ttk.Label(self, text="Train CSV:").grid(row=row, column=0, sticky="e", **pad)
        self.train_csv.grid(row=row, column=1, sticky="w", **pad)
        self.btn_train.grid(row=row, column=2, sticky="w", **pad)
        ttk.Label(self, text="Test CSV:").grid(row=row, column=3, sticky="e", **pad)
        self.test_csv.grid(row=row, column=4, sticky="w", **pad)
        self.btn_test.grid(row=row, column=5, sticky="w", **pad)

        row += 1
        ttk.Label(self, text="Nodes:").grid(row=row, column=0, sticky="e", **pad)
        self.nodes.grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(self, text="Rounds:").grid(row=row, column=2, sticky="e", **pad)
        self.rounds.grid(row=row, column=3, sticky="w", **pad)
        ttk.Label(self, text="Seed:").grid(row=row, column=4, sticky="e", **pad)
        self.seed.grid(row=row, column=5, sticky="w", **pad)

        row += 1
        ttk.Label(self, text="Topology:").grid(row=row, column=0, sticky="e", **pad)
        self.topology.grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(self, text="Partition:").grid(row=row, column=2, sticky="e", **pad)
        self.partition.grid(row=row, column=3, sticky="w", **pad)

        row += 1
        ttk.Label(self, text="Target Name:").grid(row=row, column=0, sticky="e", **pad)
        self.target_name.grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(self, text="Aggregator:").grid(row=row, column=2, sticky="e", **pad)
        self.aggregator.grid(row=row, column=3, sticky="w", **pad)
        ttk.Label(self, text="Dataset:").grid(row=row, column=4, sticky="e", **pad)
        self.dataset.grid(row=row, column=5, sticky="w", **pad)

        row += 1
        self.btn_run.grid(row=row, column=0, columnspan=2, **pad)
        self.output.grid(row=row+1, column=0, columnspan=6, **pad)

    def _browse_file(self, entry_widget):
        path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            entry_widget.delete(0, "end")
            entry_widget.insert(0, path)

    def _run_cli(self):
        def run():
            venv_python = sys.executable
            cmd = [venv_python, SCRIPT_NAME,
                   "--framework", self.framework.get()]

            data = self.data_csv.get().strip()
            if data:
                cmd += ["--data_csv", data]
            else:
                t = self.train_csv.get().strip()
                e = self.test_csv.get().strip()
                if t: cmd += ["--train_csv", t]
                if e: cmd += ["--test_csv", e]

            try:
                cmd += [
                    "--nodes", str(int(self.nodes.get())),
                    "--rounds", str(int(self.rounds.get())),
                    "--seed", str(int(self.seed.get())),
                    "--topology", self.topology.get(),
                    "--protocol", self.protocol.get(),
                    "--target_name", self.target_name.get(),
                    "--agregator", self.aggregator.get(),
                    "--partition_strategy", self.partition.get(),
                ]
            except ValueError as ve:
                self.after(0, lambda: messagebox.showerror("Invalid input", f"Number fields must be integers.\n{ve}"))
                return

            ds = self.dataset.get().strip()
            if ds:
                cmd += ["--dataset", ds]

            self.output.delete("1.0", tk.END)
            self.output.insert(tk.END, f"Running: {' '.join(cmd)}\n\n")

            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, encoding="utf-8", errors="replace")
                # Mostrar stdout y stderr en tiempo real
                while True:
                    out_line = proc.stdout.readline()
                    if out_line:
                        self.output.insert(tk.END, out_line)
                        self.output.see(tk.END)
                    # Leer stderr en paralelo
                    while True:
                        err_line = proc.stderr.readline()
                        if err_line:
                            self.output.insert(tk.END, err_line)
                            self.output.see(tk.END)
                        else:
                            break
                    if proc.poll() is not None:
                        break
                # Leer el resto de la salida
                for out_line in proc.stdout:
                    self.output.insert(tk.END, out_line)
                    self.output.see(tk.END)
                for err_line in proc.stderr:
                    self.output.insert(tk.END, err_line)
                    self.output.see(tk.END)
                self.output.insert(tk.END, f"Return code: {proc.returncode}\n")
            except Exception as e:
                self.output.insert(tk.END, f"Failed to run process: {e}")

        threading.Thread(target=run, daemon=True).start()

if __name__ == "__main__":
    app = FedCliApp()
    app.mainloop()
