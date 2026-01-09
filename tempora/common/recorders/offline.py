import textwrap

import numpy as np

from .base import Recorder


class OfflineRecorder(Recorder):
    def __init__(self):
        self.num_samples_list = []
        self.num_correct_list = []

        self.prediction_times = []  # Response latency, includes intrinsic adaptation overhead (before prediction)
        self.adaptation_times = []  # Post-hoc (extrinsic) adaptation overhead incurred after making a prediction.
        self.wall_clock_times = []

    def record(self, num_correct, num_samples, prediction_time, wall_clock_time):
        self.num_samples_list.append(num_samples)
        self.num_correct_list.append(num_correct)

        adaptation_time = max(0, wall_clock_time - prediction_time)
        self.prediction_times.append(prediction_time)
        self.wall_clock_times.append(wall_clock_time)
        self.adaptation_times.append(adaptation_time)

    def emit(self):
        summary = {
            # Counters
            "num_batches": len(self.num_samples_list),
            "num_samples": sum(self.num_samples_list),
            "num_correct": sum(self.num_correct_list),
            # Timing
            "prediction_time": {
                "avg": np.mean(self.prediction_times),
                "std": np.std(self.prediction_times),
                "min": np.min(self.prediction_times),
                "max": np.max(self.prediction_times),
            },
            "adaptation_time": {
                "avg": np.mean(self.adaptation_times),
                "std": np.std(self.adaptation_times),
                "min": np.min(self.adaptation_times),
                "max": np.max(self.adaptation_times),
            },
            "wall_clock_time": {
                "avg": np.mean(self.wall_clock_times),
                "std": np.std(self.wall_clock_times),
                "min": np.min(self.wall_clock_times),
                "max": np.max(self.wall_clock_times),
            },
            # Batch-wise metrics
            "batch_summaries": []
        }

        for i in range(len(self.num_samples_list)):
            summary['batch_summaries'].append({
                "index": i,
                "num_samples": self.num_samples_list[i],
                "num_correct": self.num_correct_list[i],
                "prediction_time": self.prediction_times[i],
                "adaptation_time": self.adaptation_times[i],
                "wall_clock_time": self.wall_clock_times[i],
            })
        
        return summary

    def print(self):
        summary = self.emit()
        
        accuracy = summary["num_correct"] / summary["num_samples"] if summary["num_samples"] > 0 else 0

        output = textwrap.dedent(f"""\
            {"-" * 32}
            Batches         : {summary["num_batches"]}
            Samples         : {summary["num_samples"]}
            Correct         : {summary["num_correct"]} ({accuracy:.2%})
            {"-" * 32}
            Prediction Time : {summary["prediction_time"]["avg"]:,.2f} ± {summary["prediction_time"]["std"]:,.2f} ms
            - Minimum       : {summary["prediction_time"]["min"]:,.2f} ms
            - Maximum       : {summary["prediction_time"]["max"]:,.2f} ms
            {"-" * 32}
            Adaptation Time : {summary["adaptation_time"]["avg"]:,.2f} ± {summary["adaptation_time"]["std"]:,.2f} ms
            - Minimum       : {summary["adaptation_time"]["min"]:,.2f} ms
            - Maximum       : {summary["adaptation_time"]["max"]:,.2f} ms
            {"-" * 32}
            Wall Clock Time : {summary["wall_clock_time"]["avg"]:,.2f} ± {summary["wall_clock_time"]["std"]:,.2f} ms
            - Minimum       : {summary["wall_clock_time"]["min"]:,.2f} ms
            - Maximum       : {summary["wall_clock_time"]["max"]:,.2f} ms
            {"-" * 32}""")

        print(output)