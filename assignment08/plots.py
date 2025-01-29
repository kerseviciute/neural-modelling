import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def get_experiment_colors(experiment):
    color_map = mpl.colormaps["Set1"]
    n_colors = len(experiment.Type.unique())
    colors = [color_map(idx / (n_colors - 1)) for idx in range(n_colors)]
    colors = dict(zip(experiment.Type.unique(), colors))
    unique_periods = experiment.Type.unique()

    return colors, unique_periods


def plot_running_score(
        experiment,
        subjects,
        ax = None,
        show_legend = True
):
    if ax is None:
        _, ax = plt.subplots()

    colors, unique_periods = get_experiment_colors(experiment)

    # Show the different experimental periods
    for i, change in experiment.iterrows():
        label = change.Type if change.Type in unique_periods else ""
        unique_periods = unique_periods[unique_periods != change.Type]

        start = change.Trial
        end = change.TrialEnd

        ax.axvspan(
            start, end,
            alpha = 0.2,
            label = label,
            color = colors.get(change.Type)
        )

    for i, subject in enumerate(subjects):
        ax.plot(
            subject.Trial,
            subject.TotalScore,
            label = f"Subject {i + 1}"
        )

    if show_legend:
        ax.legend(loc = "center left", bbox_to_anchor = (1, 0.5))

    ax.set_xlabel("Trial")
    ax.set_ylabel("Score")
    ax.set_title("Running score")


def plot_trial_score(
        experiment,
        subjects,
        ax = None
):
    if ax is None:
        _, ax = plt.subplots()

    colors, unique_periods = get_experiment_colors(experiment)

    # Show the different experimental periods
    for i, change in experiment.iterrows():
        label = change.Type if change.Type in unique_periods else ""
        unique_periods = unique_periods[unique_periods != change.Type]

        start = change.Trial
        end = change.TrialEnd

        ax.axvspan(
            start, end,
            alpha = 0.2,
            label = label,
            color = colors.get(change.Type)
        )

    subject_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, subject in enumerate(subjects):
        ax.plot(
            subject.Trial,
            subject.Score,
            label = f"Subject {i + 1}",
            alpha = 0.5,
            color = subject_colors[i]
        )

        means = subject.\
            groupby("Type").\
            mean("Score").\
            reset_index(drop = False)[["Type", "Score", "Trial"]].\
            rename(columns = {"Score": "Mean"})

        for _, mean in means.iterrows():
            ax.plot(
                [mean.Trial - 50 / 2, mean.Trial + 50 / 2], [mean.Mean, mean.Mean],
                color = subject_colors[i], linestyle = "-", linewidth = 1.5
            )

    ax.legend(loc = "center left", bbox_to_anchor = (1, 0.5))
    ax.set_xlabel("Trial")
    ax.set_ylabel("Score")
    ax.set_title("Score in each trial")


def define_experiment():
    experiment = pd.DataFrame({
        "Trial": [1, 51, 101, 151, 201],
        "TrialEnd": np.array([51, 101, 151, 201, 251]) - 1,
        "Type": ["NormalGradual", "Trajectory", "EndPos", "RL", "NormalSudden"]
    })

    n_attempts = 250

    attempts = list(range(1, n_attempts + 1))

    types = []
    for i in range(len(experiment)):
        start = experiment["Trial"][i]
        end = experiment["Trial"][i + 1] if i + 1 < len(experiment) else n_attempts + 1
        types.extend([experiment["Type"][i]] * (end - start))

    full_experiment = pd.DataFrame({ "Trial": attempts, "Type": types })

    return experiment, full_experiment
