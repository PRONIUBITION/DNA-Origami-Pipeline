"""Training Monitoring GUI
Real-time display of training progress and metrics (window version with line chart)
Usage: python monitor_gui.py"""

import os
import time
import sys
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import re

# Matplotlib embedded in Tkinter
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np


class TrainingMonitorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("""HRNet Training Monitor""")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)

        # Set Style
        self.style = ttk.Style()
        self.style.configure("Header.TLabel", font=("Microsoft YaHei", 16, "bold"))
        self.style.configure("Metric.TLabel", font=("Consolas", 12))
        self.style.configure("Status.TLabel", font=("Microsoft YaHei", 10))

        # Operating status
        self.monitoring = False
        self.run_dir = None
        self.log_file = None

        # Historical Data Storage
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.pcks = []
        self.oks_list = []

        self._create_ui()
        self.find_training_task()

    def _create_ui(self):
        """Create UI interface"""
        # Main Frame - Use PanedWindow to split left and right
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left Frame - Metrics and Logs
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)

        # Right Frame - Charts
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)

        # ===== Left: Title Area =====
        header_frame = ttk.Frame(left_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(header_frame, text="""HRNet Training Monitoring""", style="Header.TLabel").pack(
            side=tk.LEFT
        )

        self.status_label = ttk.Label(
            header_frame, text="""Searching for training tasks...""", style="Status.TLabel"
        )
        self.status_label.pack(side=tk.RIGHT)

        # ===== Left: Metrics Card Area =====
        metrics_frame = ttk.LabelFrame(left_frame, text="""Training metrics""", padding="10")
        metrics_frame.pack(fill=tk.X, pady=(0, 10))

        # Use grid layout for metrics
        ttk.Label(metrics_frame, text="""Current round""", font=("Microsoft YaHei", 10)).grid(
            row=0, column=0, padx=10
        )
        self.epoch_var = tk.StringVar(value="--")
        ttk.Label(
            metrics_frame, textvariable=self.epoch_var, style="Metric.TLabel", foreground="#0066cc"
        ).grid(row=1, column=0, padx=10)

        ttk.Label(metrics_frame, text="""Training loss""", font=("Microsoft YaHei", 10)).grid(
            row=0, column=1, padx=10
        )
        self.train_loss_var = tk.StringVar(value="--")
        ttk.Label(
            metrics_frame,
            textvariable=self.train_loss_var,
            style="Metric.TLabel",
            foreground="#cc3300",
        ).grid(row=1, column=1, padx=10)

        ttk.Label(metrics_frame, text="""Validation loss""", font=("Microsoft YaHei", 10)).grid(
            row=0, column=2, padx=10
        )
        self.val_loss_var = tk.StringVar(value="--")
        ttk.Label(
            metrics_frame,
            textvariable=self.val_loss_var,
            style="Metric.TLabel",
            foreground="#009933",
        ).grid(row=1, column=2, padx=10)

        ttk.Label(metrics_frame, text="PCK", font=("Microsoft YaHei", 10)).grid(
            row=2, column=0, pady=(10, 0)
        )
        self.pck_var = tk.StringVar(value="--")
        ttk.Label(
            metrics_frame, textvariable=self.pck_var, style="Metric.TLabel", foreground="#6600cc"
        ).grid(row=3, column=0)

        ttk.Label(metrics_frame, text="OKS", font=("Microsoft YaHei", 10)).grid(
            row=2, column=1, pady=(10, 0)
        )
        self.oks_var = tk.StringVar(value="--")
        ttk.Label(
            metrics_frame, textvariable=self.oks_var, style="Metric.TLabel", foreground="#cc6600"
        ).grid(row=3, column=1)

        ttk.Label(metrics_frame, text="""Weight File""", font=("Microsoft YaHei", 10)).grid(
            row=2, column=2, pady=(10, 0)
        )
        self.weights_var = tk.StringVar(value="--")
        ttk.Label(
            metrics_frame,
            textvariable=self.weights_var,
            style="Metric.TLabel",
            foreground="#666666",
        ).grid(row=3, column=2)

        # ===== Left: Log Display Area =====
        log_frame = ttk.LabelFrame(left_frame, text="""Latest Logs""", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.log_text = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, font=("Consolas", 9), height=10, state=tk.DISABLED
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # ===== Left: Button Area =====
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X)

        self.start_btn = ttk.Button(
            button_frame, text="""Start Monitoring""", command=self.start_monitoring
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.stop_btn = ttk.Button(
            button_frame,
            text="""Stop Monitoring""",
            command=self.stop_monitoring,
            state=tk.DISABLED,
        )
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(button_frame, text="""Refresh""", command=self.manual_refresh).pack(
            side=tk.LEFT, padx=(0, 5)
        )
        ttk.Button(button_frame, text="""Clear Chart""", command=self.clear_chart).pack(
            side=tk.LEFT, padx=(0, 5)
        )
        ttk.Button(button_frame, text="""Exit""", command=self.quit).pack(side=tk.RIGHT)

        # Bottom Information
        self.info_label = ttk.Label(
            left_frame, text="""Ready""", font=("Microsoft YaHei", 9), foreground="gray"
        )
        self.info_label.pack(fill=tk.X, pady=(5, 0))

        # ===== Right side: Chart area =====
        chart_frame = ttk.LabelFrame(right_frame, text="""Loss function curve""", padding="5")
        chart_frame.pack(fill=tk.BOTH, expand=True)

        # Create a matplotlib plot
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Epoch", fontsize=10)
        self.ax.set_ylabel("Loss", fontsize=10)
        self.ax.set_title("Training & Validation Loss", fontsize=12)
        self.ax.grid(True, alpha=0.3)

        # Create a canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, chart_frame)
        toolbar.update()

    def get_latest_run_dir(self):
        """Get the latest training directory"""
        script_dir = Path(__file__).parent.resolve()
        runs_dir = script_dir / "runs"
        if not runs_dir.exists():
            return None

        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("train_")]
        if not run_dirs:
            return None

        return sorted(run_dirs)[-1]

    def find_training_task(self):
        """Find Training Tasks"""
        self.run_dir = self.get_latest_run_dir()

        if not self.run_dir:
            self.status_label.config(text="""Training directory not found""", foreground="red")
            self.info_label.config(text="""Please start training first: python train.py""")
            return False

        self.log_file = self.run_dir / "logs" / "train.log"
        if not self.log_file.exists():
            self.status_label.config(text="""Log file not found""", foreground="red")
            self.info_label.config(text=f"""Log file does not exist: {self.log_file}""")
            return False

        self.status_label.config(text=f"Running: {self.run_dir.name}", foreground="green")
        self.info_label.config(text=f"""Log file: {self.log_file}""")
        return True

    def parse_all_history(self, content):
        """Parse all historical loss data"""
        lines = content.split("\n")

        epochs = []
        train_losses = []
        val_losses = []
        pcks = []
        oks_list = []

        current_epoch = None
        current_train_loss = None

        for line in lines:
            # Extract Epoch and Train Loss
            if "Epoch" in line and "Train Loss:" in line:
                try:
                    match = re.search(r"Epoch\s+(\d+)/(\d+)\s+-\s+Train Loss:\s+([\d.]+)", line)
                    if match:
                        epoch_num = int(match.group(1))
                        current_epoch = epoch_num
                        current_train_loss = float(match.group(3))
                except:
                    pass

            # Extract validation information (corresponding to the training epoch)
            if "Validation - Loss:" in line and current_epoch is not None:
                try:
                    match = re.search(r"Loss:\s+([\d.]+)", line)
                    if match:
                        val_loss = float(match.group(1))

                        # Save data
                        epochs.append(current_epoch)
                        train_losses.append(current_train_loss)
                        val_losses.append(val_loss)

                        # Extract PCK and OKS
                        pck_match = re.search(r"PCK:\s+([\d.]+)", line)
                        if pck_match:
                            pcks.append(float(pck_match.group(1)))
                        else:
                            pcks.append(None)

                        oks_match = re.search(r"OKS:\s+([\d.]+)", line)
                        if oks_match:
                            oks_list.append(float(oks_match.group(1)))
                        else:
                            oks_list.append(None)

                        current_epoch = None
                        current_train_loss = None
                except:
                    pass

        return epochs, train_losses, val_losses, pcks, oks_list

    def parse_latest_info(self, content):
        """Parse the latest information (used to display the current indicators)"""
        info = {}
        lines = content.split("\n")

        for line in reversed(lines):
            if "Epoch" in line and "Train Loss:" in line:
                try:
                    match = re.search(r"Epoch\s+(\d+/\d+)\s+-\s+Train Loss:\s+([\d.]+)", line)
                    if match and "epoch" not in info:
                        info["epoch"] = match.group(1)
                        info["train_loss"] = match.group(2)
                except:
                    pass

            if "Validation - Loss:" in line and "val_loss" not in info:
                try:
                    match = re.search(r"Loss:\s+([\d.]+)", line)
                    if match:
                        info["val_loss"] = match.group(1)

                    match = re.search(r"PCK:\s+([\d.]+)", line)
                    if match:
                        info["pck"] = match.group(1)

                    match = re.search(r"OKS:\s+([\d.]+)", line)
                    if match:
                        info["oks"] = match.group(1)
                except:
                    pass

            if len(info) >= 5:
                break

        return info, lines[-20:] if len(lines) > 20 else lines

    def update_chart(self):
        """Update line chart"""
        if not self.epochs:
            return

        self.ax.clear()

        # Plot training loss
        if self.epochs and self.train_losses:
            self.ax.plot(
                self.epochs,
                self.train_losses,
                "b-",
                linewidth=2,
                label="Train Loss",
                marker="o",
                markersize=4,
            )

        # Plot validation loss
        if self.epochs and self.val_losses:
            self.ax.plot(
                self.epochs,
                self.val_losses,
                "r-",
                linewidth=2,
                label="Val Loss",
                marker="s",
                markersize=4,
            )

        self.ax.set_xlabel("Epoch", fontsize=10)
        self.ax.set_ylabel("Loss", fontsize=10)
        self.ax.set_title("Training & Validation Loss", fontsize=12)
        self.ax.legend(loc="upper right")
        self.ax.grid(True, alpha=0.3)

        # Automatically adjust the Y-axis range
        if self.train_losses and self.val_losses:
            all_losses = [l for l in self.train_losses + self.val_losses if l is not None]
            if all_losses:
                ymin = min(all_losses) * 0.9
                ymax = max(all_losses) * 1.1
                self.ax.set_ylim(ymin, ymax)

        # Automatically adjust the X-axis range
        if self.epochs:
            self.ax.set_xlim(0, max(self.epochs) + 1)

        self.canvas.draw()

    def update_display(self):
        """Update Display"""
        if not self.log_file or not self.log_file.exists():
            return

        try:
            with open(self.log_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Update historical data (for charts)
            self.epochs, self.train_losses, self.val_losses, self.pcks, self.oks_list = (
                self.parse_all_history(content)
            )

            # Update the latest indicators
            info, recent_lines = self.parse_latest_info(content)

            if "epoch" in info:
                self.epoch_var.set(info["epoch"])
            if "train_loss" in info:
                self.train_loss_var.set(info["train_loss"])
            if "val_loss" in info:
                self.val_loss_var.set(info["val_loss"])
            if "pck" in info:
                self.pck_var.set(info["pck"])
            if "oks" in info:
                self.oks_var.set(info["oks"])

            # Number of weight files updated
            weights_dir = self.run_dir / "weights"
            if weights_dir.exists():
                weight_files = list(weights_dir.glob("*.pth"))
                self.weights_var.set(f"""{len(weight_files)} items""")
            else:
                self.weights_var.set("""0""")

            # Update log display
            self.log_text.config(state=tk.NORMAL)
            self.log_text.delete(1.0, tk.END)
            self.log_text.insert(tk.END, "\n".join(recent_lines))
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)

            # Update chart
            self.update_chart()

            # Update Time
            self.info_label.config(text=f"""Last updated: {datetime.now().strftime("%H:%M:%S")}""")

        except Exception as e:
            self.info_label.config(text=f"""Update failed: {e}""", foreground="red")

    def clear_chart(self):
        """Clear chart data"""
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.pcks = []
        self.oks_list = []
        self.ax.clear()
        self.ax.set_xlabel("Epoch", fontsize=10)
        self.ax.set_ylabel("Loss", fontsize=10)
        self.ax.set_title("Training & Validation Loss", fontsize=12)
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def monitoring_loop(self):
        """Monitoring loop"""
        while self.monitoring:
            self.root.after(0, self.update_display)
            time.sleep(2)

    def start_monitoring(self):
        """Start Monitoring"""
        if not self.run_dir:
            if not self.find_training_task():
                return

        self.monitoring = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        self.monitor_thread.start()

        # Update once now
        self.update_display()

    def stop_monitoring(self):
        """Stop Monitoring"""
        self.monitoring = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def manual_refresh(self):
        """Manual refresh"""
        if not self.run_dir:
            self.find_training_task()
        self.update_display()

    def quit(self):
        """Exit program"""
        self.monitoring = False
        # Save the loss function curve to the training folder
        if self.run_dir and self.run_dir.exists() and self.fig:
            save_path = self.run_dir / f"loss_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            try:
                self.fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
                print(f"""Chart saved to: {save_path}""")
            except Exception as e:
                print(f"""Failed to save chart: {e}""")
        plt.close(self.fig)
        self.root.quit()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = TrainingMonitorGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.quit)
    root.mainloop()


if __name__ == "__main__":
    main()
