import math


class LogstdScheduler:
    
    def __init__(
        self,
        init_logstd: float = -1.0,
        final_logstd: float = -5.0,
        warmup_steps: int = 100000,
        decay_steps: int = 900000,
        schedule_type: str = "linear",
    ):
        self.init_logstd = init_logstd
        self.final_logstd = final_logstd
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.schedule_type = schedule_type

    def get_logstd(self, step: int) -> float:
        
        if self.schedule_type == "fixed":
            return self.init_logstd

        if step < self.warmup_steps:
            return self.init_logstd

        decay_t = step - self.warmup_steps
        progress = min(1.0, decay_t / max(1, self.decay_steps))

        if self.schedule_type == "linear":
            return self.init_logstd + progress * (self.final_logstd - self.init_logstd)

        elif self.schedule_type == "cosine":

            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return self.final_logstd + cosine_factor * (self.init_logstd - self.final_logstd)

        elif self.schedule_type == "exponential":











            if self.init_logstd == 0.0 or self.final_logstd == 0.0:

                return self.init_logstd + progress * (self.final_logstd - self.init_logstd)
            ratio = self.final_logstd / self.init_logstd
            if ratio <= 0.0:

                return self.init_logstd + progress * (self.final_logstd - self.init_logstd)
            return self.init_logstd * (ratio ** progress)

        return self.init_logstd


def main() -> None:
    
    import os

    import numpy as np


    import matplotlib.pyplot as plt

    warmup_steps = 100_000
    decay_steps = 900_000
    total_steps = warmup_steps + decay_steps

    init_logstd = -1.0
    final_logstd = -5.0

    steps = np.arange(total_steps + 1, dtype=np.int32)

    schedule_types = ("linear", "cosine", "exponential")
    curves: dict[str, np.ndarray] = {}

    for schedule_type in schedule_types:
        sched = LogstdScheduler(
            init_logstd=init_logstd,
            final_logstd=final_logstd,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            schedule_type=schedule_type,
        )
        curves[schedule_type] = np.asarray([sched.get_logstd(int(s)) for s in steps], dtype=np.float32)

    plt.figure(figsize=(10, 5))
    for schedule_type in schedule_types:
        plt.plot(steps, curves[schedule_type], label=schedule_type)

    plt.axvline(
        warmup_steps,
        linestyle="--",
        linewidth=1,
        color="gray",
        alpha=0.7,
        label="warmup_end",
    )
    plt.title(
        f"logstd schedules (init={init_logstd}, final={final_logstd}, warmup={warmup_steps}, decay={decay_steps})"
    )
    plt.xlabel("step")
    plt.ylabel("logstd")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_name = "logstd_schedules.png"
    out_path = os.path.join(os.getcwd(), out_name)
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[std_scheduler] Saved plot to: {out_path}")


if __name__ == "__main__":
    main()