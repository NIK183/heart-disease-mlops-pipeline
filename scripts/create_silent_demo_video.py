"""Create a silent evidence-based demo video from real project screenshots.

This is intentionally a captioned walkthrough, not a simulation of a live run.
All visual evidence comes from the project figures and real screenshots captured
during verification.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_VIDEO_PACKAGES = PROJECT_ROOT / ".tools" / "video"
if LOCAL_VIDEO_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_VIDEO_PACKAGES))

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont, ImageOps  # noqa: E402

WIDTH, HEIGHT, FPS = 1920, 1080, 30
OUTPUT = PROJECT_ROOT / "heart_disease_mlops_demo_silent.mp4"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(Path("C:/Windows/Fonts") / name, size=size)


def fit(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    canvas = Image.new("RGB", (WIDTH, HEIGHT), "#101827")
    contained = ImageOps.contain(image, (WIDTH - 100, HEIGHT - 180))
    x = (WIDTH - contained.width) // 2
    y = 120 + (HEIGHT - 180 - contained.height) // 2
    canvas.paste(contained, (x, y))
    return canvas


def titled_frame(title: str, subtitle: str, source: Path | None = None) -> Image.Image:
    frame = fit(Image.open(source)) if source else Image.new("RGB", (WIDTH, HEIGHT), "#101827")
    draw = ImageDraw.Draw(frame)
    draw.rectangle((0, 0, WIDTH, 108), fill="#0b1220")
    draw.text((48, 24), title, fill="white", font=font(38, bold=True))
    draw.rectangle((0, HEIGHT - 72, WIDTH, HEIGHT), fill="#0b1220")
    draw.text((48, HEIGHT - 52), subtitle, fill="#b9c7d9", font=font(25))
    return frame


def title_card(title: str, lines: list[str]) -> Image.Image:
    frame = Image.new("RGB", (WIDTH, HEIGHT), "#101827")
    draw = ImageDraw.Draw(frame)
    draw.text((110, 180), title, fill="#60a5fa", font=font(62, bold=True))
    y = 330
    for line in lines:
        draw.text((135, y), line, fill="white", font=font(34))
        y += 75
    draw.text((135, 900), "Silent evidence-based walkthrough | AIMLCZG523 MLOps Assignment 01",
              fill="#b9c7d9", font=font(26))
    return frame


def add_clip(writer: imageio.Writer, frame: Image.Image, seconds: int) -> None:
    data = np.asarray(frame)
    for _ in range(seconds * FPS):
        writer.append_data(data)


def main() -> None:
    figures = PROJECT_ROOT / "reports" / "figures"
    shots = PROJECT_ROOT / "reports" / "screenshots"
    clips: list[tuple[str, str, Path | None, int]] = [
        ("Heart Disease MLOps Pipeline", "End-to-end, reproducible, containerized and monitored ML workflow", None, 5),
        ("Data acquisition and EDA", "Class balance from the cleaned UCI heart-disease data", figures / "class_balance.png", 5),
        ("Data acquisition and EDA", "Correlation analysis used to understand relationships in clinical features", figures / "correlation_heatmap.png", 5),
        ("Model development", "Final Logistic Regression ROC curve; candidate models were selected using nested cross-validation", figures / "roc_curve_logisticregression.png", 5),
        ("Experiment tracking", "MLflow records the three completed candidate-model runs", shots / "03_mlflow_experiment_runs.png", 6),
        ("Experiment tracking", "Nested-CV and held-out test metrics recorded in MLflow", shots / "03b_mlflow_run_metrics.png", 6),
        ("Experiment tracking", "Best parameters and GridSearchCV nested-CV protocol", shots / "03b_mlflow_run_parameters.png", 6),
        ("CI/CD", "GitHub Actions: lint, training/evaluation and Docker verification all passed", shots / "05_github_actions_run.png", 5),
        ("Model serving API", "FastAPI Swagger documentation exposes health, metrics and predict endpoints", shots / "06_api_swagger_docs.png", 5),
        ("Model serving API", "Live /predict request returns binary prediction and confidence", shots / "06b_api_predict_response.png", 6),
        ("Container verification", "Automated serving verification confirms health and inference response", shots / "06c_serving_verification.png", 5),
        ("Container verification", "Docker build and isolated API smoke test succeeded", shots / "06d_docker_build_run.png", 5),
        ("Kubernetes deployment", "Two running API pods, LoadBalancer service, HPA and Ingress", shots / "07_kubernetes_deployment.png", 6),
        ("Prometheus monitoring", "Prometheus target is UP and scraping the API metrics endpoint", shots / "08_prometheus_targets.png", 5),
        ("Prometheus monitoring", "Prediction counter is grouped by predicted outcome", shots / "08c_prometheus_predictions_by_outcome.png", 5),
        ("Logging", "Structured API request logs captured from the running monitoring stack", shots / "08e_api_request_logs.png", 5),
        ("Grafana dashboard", "Grafana displays request rate, p95 latency and prediction outcomes", shots / "08f_grafana_dashboard.png", 6),
    ]
    with imageio.get_writer(OUTPUT, fps=FPS, codec="libx264", quality=8, macro_block_size=None) as writer:
        for title, subtitle, source, duration in clips:
            if source is not None and not source.exists():
                raise FileNotFoundError(f"Required demo evidence is missing: {source}")
            frame = titled_frame(title, subtitle, source) if source else title_card(title, [subtitle])
            add_clip(writer, frame, duration)
        add_clip(writer, title_card("Thank you", ["Repository: github.com/NIK183/heart-disease-mlops-pipeline", "See README.md for clean setup and local access instructions."]), 5)
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    main()
