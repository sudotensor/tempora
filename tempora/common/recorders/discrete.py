import textwrap

import numpy as np

from .base import Recorder


class DiscreteRecorder(Recorder):
    def __init__(self):
        self.num_correct_list = []
        self.num_samples_list = []
        self.num_skipped_list = []

        self.start_timestamps = []  # Simulated timestamp when processing started
        self.prediction_times = []  # Time to make a prediction once started, includes intrinsic adaptation overhead
        self.adaptation_times = []  # Post-hoc (extrinsic) adaptation overhead incurred after making a prediction.
        self.wall_clock_times = []
        self.queue_wait_times = []  # Time spent waiting to start; effective latency = prediction_time + queue_wait_time

    def record(self, num_correct, num_samples, num_skipped, start_timestamp, prediction_time, wall_clock_time, queue_wait_time):
        self.num_correct_list.append(num_correct)
        self.num_samples_list.append(num_samples)
        self.num_skipped_list.append(num_skipped)
        
        if num_skipped == 0:  # Processed batch
            adaptation_time = max(0, wall_clock_time - prediction_time)
            self.start_timestamps.append(start_timestamp)
            self.prediction_times.append(prediction_time)
            self.wall_clock_times.append(wall_clock_time)
            self.adaptation_times.append(adaptation_time)
            self.queue_wait_times.append(queue_wait_time)
            
    def emit(self):
        summary = {
            # Counters
            "num_batches": len(self.num_samples_list),
            "num_samples": sum(self.num_samples_list),
            "num_correct": sum(self.num_correct_list),
            "num_skipped": sum(self.num_skipped_list),
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
            "queue_wait_time": {
                "avg": np.mean(self.queue_wait_times),
                "std": np.std(self.queue_wait_times),
                "min": np.min(self.queue_wait_times),
                "max": np.max(self.queue_wait_times),
            },
            # Batch-wise metrics
            "batch_summaries": [],
        }

        offset = 0
        for i in range(len(self.num_samples_list)):
            summary["batch_summaries"].append({
                "index": i,
                "num_samples": self.num_samples_list[i],
                "num_correct": self.num_correct_list[i],
                "num_skipped": self.num_skipped_list[i],
                "start_timestamp": self.start_timestamps[i - offset] if self.num_skipped_list[i] == 0 else 0,
                "prediction_time": self.prediction_times[i - offset] if self.num_skipped_list[i] == 0 else 0,
                "adaptation_time": self.adaptation_times[i - offset] if self.num_skipped_list[i] == 0 else 0,
                "wall_clock_time": self.wall_clock_times[i - offset] if self.num_skipped_list[i] == 0 else 0,
                "queue_wait_time": self.queue_wait_times[i - offset] if self.num_skipped_list[i] == 0 else 0,
            })
            offset += int(self.num_skipped_list[i] != 0)

        return summary

    def print(self):
        summary = self.emit()

        availability = (1 - summary["num_skipped"] / summary["num_samples"]) if summary["num_samples"] > 0 else 0
        system_accuracy = summary["num_correct"] / summary["num_samples"] if summary["num_samples"] > 0 else 0
        served_accuracy = system_accuracy / availability if availability > 0 else 0

        waitspan = 0
        for i in range(1, len(self.start_timestamps)):
            waitspan += max(0, self.start_timestamps[i] - self.start_timestamps[i - 1] - self.wall_clock_times[i - 1])
        makespan = sum(self.wall_clock_times) + waitspan

        output = textwrap.dedent(f"""\
            {"-" * 32}
            Batches         : {summary["num_batches"]}
            Samples         : {summary["num_samples"]}
            Correct         : {summary["num_correct"]} (System: {system_accuracy:.2%}, Served: {served_accuracy:.2%})
            Skipped         : {summary["num_skipped"]} (Availability: {availability:.2%})
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
            {"-" * 32}
            Queue Wait Time : {summary["queue_wait_time"]["avg"]:,.2f} ± {summary["queue_wait_time"]["std"]:,.2f} ms
            - Minimum       : {summary["queue_wait_time"]["min"]:,.2f} ms
            - Maximum       : {summary["queue_wait_time"]["max"]:,.2f} ms
            {"-" * 32}
            Makespan        : {makespan:,.2f} ms
            Waitspan        : {waitspan:,.2f} ms
            {"-" * 32}""")

        print(output)
