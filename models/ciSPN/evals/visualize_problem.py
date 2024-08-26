import numpy as np
import torch

from ciSPN.datasets.particleCollisionDataset import attr_to_index
from external.GalaxyCollision.data_handler import sample_simulation as gc_sample
from external.GalaxyCollision.data_handler import (
    continue_simulation as gc_sample_cont,
)
from external.GalaxyCollision.data_handler import visualize_sample as gc_visualize
from external.particles_in_a_box.data_handler import sample_data as pc_sample
from external.particles_in_a_box.data_handler import visualize_sample as pc_visualize


def visualize_problem(dataset, eval_wrapper, output_dir, parameters, vis_arguments):
    if dataset == "PC":
        visualize_pc(eval_wrapper, output_dir, parameters, vis_arguments)
    elif dataset == "GC":
        visualize_gc(eval_wrapper, output_dir, parameters, vis_arguments)
    else:
        print(f"No visualization implemented for dataset {dataset}")


def visualize_pc(eval_wrapper, output_dir, parameters, vis_arguments):
    # visualize an example for the particle collision dataset
    device, num_variables, placeholder_target_batch, marginalized = parameters
    if vis_arguments == "":
        test_example = 0
    else:
        test_example = int(vis_arguments)

    if test_example == 0:
        seed = 0
        int_time = 25
        int_part = 0
        int_attr = "x"
        int_value = 2.5
    elif test_example == 1:
        seed = 99
        int_time = 0
        int_part = 2
        int_attr = "vy"
        int_value = 5
    else:
        raise RuntimeError(f"No example defined for test_example=={test_example}")

    # simulation without intervention
    original = pc_sample(seed, 50)

    # simulation with intervention (ground truth, counterfactual to non-interventional sample above)
    expectation = pc_sample(
        seed, 50, intervention_info=(int_time, int_part, int_attr, int_value)
    )
    expectation = torch.Tensor(expectation)

    # here: calculate model prediction
    prediction = torch.zeros_like(expectation)
    prediction[0] = expectation[0]
    current_time_step = prediction[0:1]
    for i in range(50):
        # get the current state (particle information) into a useful shape
        current_state = [
            [current_time_step[0, j * 4 + k] for k in range(4)]
            for j in range(current_time_step.shape[1] // 4)
        ]
        if i == int_time:
            # intervention
            next_time_step = torch.Tensor(
                pc_sample(
                    seed,
                    n_frames=1,
                    intervention_info=(0, int_part, int_attr, int_value),
                    continue_data=current_state,
                )[1:2]
            )
            intervention_vector = torch.zeros((1, num_variables))
            index = int_part * 4 + attr_to_index[int_attr]
            intervention_vector[0, index] = 1
            condition = torch.concat(
                (current_time_step, intervention_vector, torch.Tensor([[int_value]])),
                dim=1,
            )
        else:
            # no intervention
            next_time_step = torch.Tensor(
                pc_sample(seed, n_frames=1, continue_data=current_state)[1:2]
            )
            condition = torch.concat(
                (
                    current_time_step,
                    torch.zeros((1, num_variables)),
                    torch.Tensor([[0]]),
                ),
                dim=1,
            )
        pred = eval_wrapper.predict(
            condition.to(device), placeholder_target_batch[0:1], marginalized[0:1]
        )
        prediction[i + 1] = pred
        current_time_step = next_time_step

    expectation = expectation.numpy()
    prediction = prediction.numpy()
    pc_visualize(
        0,
        original,
        n_particles=None,
        box_size=None,
        radii_limits=None,
        output_dir=output_dir,
    )
    pc_visualize(
        0,
        expectation,
        n_particles=None,
        box_size=None,
        radii_limits=None,
        output_dir=output_dir,
    )
    pc_visualize(
        0,
        prediction,
        n_particles=None,
        box_size=None,
        radii_limits=None,
        output_dir=output_dir,
    )


def visualize_gc(eval_wrapper, output_dir, parameters, vis_arguments):
    # visualize an example for the galaxy collision dataset
    device, num_variables, placeholder_target_batch, marginalized, method = parameters
    if vis_arguments == "":
        seed = 42
        nr_part_vis = 200
        time_steps = 100
    else:
        seed, nr_part_vis, time_steps = vis_arguments.split("-")
        seed = int(seed)
        nr_part_vis = int(nr_part_vis)
        time_steps = int(time_steps)

    # generate a simulation (ground truth)
    data, nr_particles = gc_sample(
        time_steps, method=method, black_holes=2, seed=seed, return_nr_particles=True
    )

    rpg = np.random.RandomState(seed)
    nr_int = 5
    intervention_particles = rpg.choice(nr_particles, nr_int, replace=False)

    def create_interventions(int_particles):
        return [(part, 0, 1) for part in int_particles]  # intervene on value 0 with +1

    interventions = create_interventions(intervention_particles)
    int_time = (
        time_steps // 2
    )  # both the time to intervene and how many timesteps follow after

    state_for_intervention = np.copy(
        data[int_time * nr_particles : (int_time + 1) * nr_particles, :num_variables]
    )
    for int_part, int_var, int_add in interventions:
        state_for_intervention[int_part, int_var] += int_add
    data_with_ints = gc_sample_cont(
        int_time, method=method, seed=42, initial_state=state_for_intervention
    )
    # comment the following line for ground truth without intervention
    data[int_time * nr_particles :] = data_with_ints

    # if nr_part_vis is smaller than the actual number of particles (=stars), only that number of stars are used and
    # shown, both for ground truth and prediction (useful if computation should not take too much time)
    data_per_state = nr_part_vis if nr_part_vis < nr_particles else nr_particles
    if nr_part_vis < nr_particles:
        # choose only some out of the full set of particles returned by gc_samples
        particles = [list(rpg.choice(range(nr_particles), nr_part_vis, replace=False))]
        # add intervention particles to be present at all times
        # for this, first remove then from if they were randomly chosen
        for particle in intervention_particles:
            if particle in particles[0]:
                particles[0].remove(particle)
        # then append them at the start (now they are definitely present once)
        particles[0] = list(intervention_particles) + particles[0]
        # if necessary, remove some other randomly chosen particles
        particles[0] = np.array(particles[0][:nr_part_vis])
        for i in range(1, time_steps):
            # calculate and include these same particles for all time steps
            particles.append(i * nr_particles + particles[0])
        # get data for those time steps
        data = data[np.array(particles).flatten()]
        intervention_particles = np.arange(
            nr_int
        )  # since they are now the first indices
    interventions = create_interventions(intervention_particles)

    gt = []  # ground truth
    initial = []
    for ts in range(time_steps):
        ts_from = data_per_state * ts
        ts_to = data_per_state * (ts + 1)
        # use median to decide for galaxy positions (should not matter for the ground truth)
        galaxies_x = np.array(
            [np.median(data[ts_from:ts_to, 4]), np.median(data[ts_from:ts_to, 8])]
        )
        galaxies_y = np.array(
            [np.median(data[ts_from:ts_to, 5]), np.median(data[ts_from:ts_to, 9])]
        )
        stars_x = data[ts_from:ts_to, 0]
        stars_y = data[ts_from:ts_to, 1]
        if ts == 0:
            initial = [galaxies_x, galaxies_y, stars_x, stars_y]
        gt.append([galaxies_x, galaxies_y, stars_x, stars_y])
    # prediction of last state is the last timestep
    gc_visualize(
        gt, output_dir / "expectation", highlight_particles=intervention_particles
    )

    preds = [initial]

    # set the initial input
    condition_np = np.zeros((data_per_state, num_variables * 2 + 1))
    condition_np[:, :num_variables] = data[0:data_per_state, :num_variables]

    for ts in range(1, time_steps):  # from 1 as the initial one is already done
        # create new placeholder and marginalized variables in case the batch is larger than 1000 instances
        pht = placeholder_target_batch[0].repeat((data_per_state, 1))
        marg = marginalized[0].repeat((data_per_state, 1))

        # predict next time step
        condition = torch.Tensor(condition_np)
        if ts == int_time:
            for int_part, int_var, int_add in interventions:
                condition[int_part, -1] = int_add
                condition[num_variables + int_var] = 1
        pred = eval_wrapper.predict(condition.to(device), pht, marg)
        pred = pred.detach().cpu().numpy()
        if ts == int_time:
            condition[:, num_variables:] = 0

        # use median to decide for galaxy positions
        galaxies_x = np.array([np.median(pred[:, 4]), np.median(pred[:, 8])])
        galaxies_y = np.array([np.median(pred[:, 5]), np.median(pred[:, 9])])
        stars_x = pred[:, 0]
        stars_y = pred[:, 1]
        preds.append([galaxies_x, galaxies_y, stars_x, stars_y])

        # prepare input for next time step
        condition_np[:, :num_variables] = pred

        # but only use the median black holes position and velocity (to keep it the same among particles)
        condition_np[:, 4:12] = np.median(pred[:, 4:12], axis=0)
    gc_visualize(
        preds, output_dir / "prediction", highlight_particles=intervention_particles
    )
