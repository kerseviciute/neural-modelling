import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import dimensions as dims
import copy


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


def plot_experiment_layout(experiment, ax):
    colors, unique_periods = get_experiment_colors(experiment)
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
    return ax


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

        means = subject. \
            groupby("Type"). \
            mean("Score"). \
            reset_index(drop = False)[["Type", "Score", "Trial"]]. \
            rename(columns = { "Score": "Mean" })

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
        "Trial": [1, 51, 101, 151],
        "TrialEnd": np.array([51, 101, 151, 201]) - 1,
        "Type": ["No motor noise", "Small motor noise", "Medium motor noise", "Large motor noise"]
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


def plot_ellipse_around_points(x, y, ax, color, nstd = 2, **kwargs):
    cov = np.cov(x, y)
    pos = (np.mean(x), np.mean(y))

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * nstd * np.sqrt(eigvals)

    ellipse = Ellipse(xy = pos, width = width, height = height, angle = angle, **kwargs)

    ax.scatter(pos[0], pos[1], marker = "x", color = color)
    ax.add_patch(ellipse)


def plot_throw_perturbation(subject):
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    min_pos_x = subject.EndPosX.min() - 100
    max_pos_x = dims.screen_width + 50

    min_pos_y = subject.EndPosY.min() - 100
    max_pos_y = dims.screen_height + 50

    fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (11, 6))

    for (i, feedback), ax in zip(enumerate(subject.Type.unique()), axs.flatten()):
        screen_ = copy.deepcopy(dims.screen)
        table_ = copy.deepcopy(dims.table)

        ax.add_patch(screen_)
        ax.add_patch(table_)
        ax.fill(dims.green_x, dims.green_y, color = "green", alpha = 0.1)
        ax.fill(dims.red_x, dims.red_y, color = "red", alpha = 0.1)

        group_data = subject[subject.Type == feedback].reset_index(drop = True).sort_values(by = "Trial")
        no_perturb_before = group_data.loc[:9]
        no_perturb_before = no_perturb_before[no_perturb_before.EndPosY < 800]

        perturb = group_data.loc[10:39]
        perturb = perturb[perturb.EndPosY < 800]

        no_perturb_after = group_data.loc[40:]
        no_perturb_after = no_perturb_after[no_perturb_after.EndPosY < 800]

        ax.scatter(
            no_perturb_before.EndPosX,
            no_perturb_before.EndPosY,
            label = "No Perturbation (Before)",
            c = default_colors[0],
            s = 10
        )
        plot_ellipse_around_points(
            no_perturb_before.EndPosX, no_perturb_before.EndPosY,
            ax = ax,
            color = default_colors[0],
            alpha = 0.75,
            edgecolor = default_colors[0],
            facecolor = "none",
            lw = 2
        )

        ax.scatter(
            perturb.EndPosX,
            perturb.EndPosY,
            label = "Perturbation",
            c = default_colors[1],
            s = 10
        )
        plot_ellipse_around_points(
            perturb.EndPosX, perturb.EndPosY,
            ax = ax,
            color = default_colors[1],
            alpha = 0.75,
            edgecolor = default_colors[1],
            facecolor = "none",
            lw = 2
        )

        ax.scatter(
            no_perturb_after.EndPosX,
            no_perturb_after.EndPosY,
            label = "No Perturbation (After)",
            c = default_colors[2],
            s = 10
        )
        plot_ellipse_around_points(
            no_perturb_after.EndPosX, no_perturb_after.EndPosY,
            ax = ax,
            color = default_colors[2],
            alpha = 0.75,
            edgecolor = default_colors[2],
            facecolor = "none",
            lw = 2
        )

        ax.set_xlim(min_pos_x, max_pos_x)
        ax.set_ylim(min_pos_y, max_pos_y)

        ax.set_title(feedback)
        ax.invert_yaxis()

    axs.flatten()[-1].legend(loc = "center left", bbox_to_anchor = (1, 0.5))

    fig.tight_layout()


def plot_throw_positions(subject):
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    min_pos_x = subject.EndPosX.min() - 100
    max_pos_x = dims.screen_width + 50

    min_pos_y = subject.EndPosY.min() - 100
    max_pos_y = dims.screen_height + 50

    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 4))

    axs[0].add_patch(copy.deepcopy(dims.screen))
    axs[1].add_patch(copy.deepcopy(dims.screen))

    axs[0].add_patch(copy.deepcopy(dims.table))
    axs[1].add_patch(copy.deepcopy(dims.table))

    axs[0].fill(dims.green_x, dims.green_y, color = "green", alpha = 0.1)
    axs[1].fill(dims.green_x, dims.green_y, color = "green", alpha = 0.1)

    axs[0].fill(dims.red_x, dims.red_y, color = "red", alpha = 0.1)
    axs[1].fill(dims.red_x, dims.red_y, color = "red", alpha = 0.1)

    for i, feedback in enumerate(subject.Type.unique()):
        group_data = subject[subject.Type == feedback]
        group_data = group_data[group_data.EndPosY < 800]
        plot_ellipse_around_points(
            group_data.EndPosX, group_data.EndPosY,
            ax = axs[0],
            color = default_colors[i],
            alpha = 0.75,
            edgecolor = default_colors[i],
            facecolor = "none",
            lw = 2,
            label = feedback
        )

    for i, feedback in enumerate(subject.Type.unique()):
        group_data = subject[subject.Type == feedback]
        axs[1].scatter(
            group_data.EndPosX,
            group_data.EndPosY,
            color = default_colors[i],
            s = 5,
            label = feedback
        )

    axs[0].set_xlim(min_pos_x, max_pos_x)
    axs[0].set_ylim(min_pos_y, max_pos_y)

    axs[1].set_xlim(min_pos_x, max_pos_x)
    axs[1].set_ylim(min_pos_y, max_pos_y)

    axs[0].invert_yaxis()
    axs[1].invert_yaxis()

    axs[1].legend(loc = "center left", bbox_to_anchor = (1, 0.5))

    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")

    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")

    axs[0].set_title("Distribution around mean final position")
    axs[1].set_title("Final pint position")

    fig.tight_layout()
