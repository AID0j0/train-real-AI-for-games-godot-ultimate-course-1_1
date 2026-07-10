import argparse
import os
import random
import time
from pathlib import Path
try: from distutils.util import strtobool
except: from setuptools._distutils.util import strtobool
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import imageio.v2 as imageio
import GPUtil
import psutil

from godot_rl.wrappers.clean_rl_wrapper import CleanRLGodotEnv


# --- CONFIGURATION ---
STACK_SIZE = 4

SAVE_PNGS = False
png_counter = 1
DEBUG_IMAGE_DIR = Path("vizualize_ai_inputs")
if SAVE_PNGS:
    DEBUG_IMAGE_DIR.mkdir(exist_ok=True)

class ImprovedGodotEnv(CleanRLGodotEnv):
    def set_curriculum(self, level: int, nums: list, strings: list):
        '''you can adjust the curriculum_payload as you see fit however then you also have to adjust the call in here
        as well as the godot update_curriculum function -> however you can theoretically put in what you want in the
        stats and tags lists'''
        curriculum_payload = {
            "level": level,
            "stats": nums,
            "tags": strings
        }
        message = {
            "type": "call",
            "method": "update_curriculum",
            "args": [curriculum_payload]
        }
        for env in self.envs:
            env._send_as_json(message)
            env._get_json_dict()

def append_resume_step(step: int):
    # requires wandb to be imported and a run to be active
    raw = wandb.run.config.get("interrupted_training_at_steps", "")
    if raw:
        existing = [s.strip() for s in raw.split(",") if s.strip().replace("_", "").isdigit()]
    else:
        existing = []
    existing.append(f"{step:_}")
    wandb.config.update(
        {"interrupted_training_at_steps": ", ".join(existing)},
        allow_val_change=True
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default=os.path.basename(__file__).rstrip(".py"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch_deterministic", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--env_path", type=str, default=None)
    parser.add_argument("--total_timesteps", type=int, default=230_000_000)
    parser.add_argument("--learning_rate", type=float, default=2.5e-4)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=171)
    parser.add_argument("--anneal_lr", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--num_minibatches", type=int, default=4)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--norm_adv", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--clip_coef", type=float, default=0.1)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--wandb_project_name", type=str, default="TEST_SpaceInvader")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a .pth checkpoint file to resume training from")
    parser.add_argument("--save_onnx", type=lambda x: bool(strtobool(x)), default=False,
                        help="Export model as .onnx after training completes") # onnx will be taken care of correctly in future courses

    args = parser.parse_args()


    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def hex_to_ndarray(hex_data, shape=(84, 84, 1)):
    """Decodes hex string from Godot or returns raw array if already decoded."""
    if isinstance(hex_data, (str, np.str_)):
        byte_data = bytes.fromhex(str(hex_data))
        arr = np.frombuffer(byte_data, dtype=np.uint8).astype(np.float32)
        picture = arr.reshape(shape)
        if SAVE_PNGS: save_game_screenshoots(picture)
        return picture
    return np.array(hex_data).astype(np.float32)


def save_game_screenshoots(picture):
    """saves singular and stacked images to a folder for inspection"""
    global png_counter
    number_of_images = 100
    if png_counter <= number_of_images:
        if png_counter == 42:
            np.set_printoptions(threshold=np.inf, linewidth=np.inf, legacy='1.13')
            print("\n\n --- printing the numbers of 1 image ---")
            # print(picture.squeeze().astype(np.uint8))
            for row in picture.squeeze().astype(np.uint8):
                print(''.join(f'{int(v):3}' for v in row))
            np.set_printoptions(threshold=1000)
        save_path = DEBUG_IMAGE_DIR / f"img_{png_counter}.png"
        imageio.imwrite(save_path, picture.squeeze().astype(np.uint8))
    elif png_counter == number_of_images + 1:
        for start in range(1, number_of_images-STACK_SIZE):
            indices = range(start, start + STACK_SIZE)
            imgs = [imageio.imread(DEBUG_IMAGE_DIR / f"img_{i}.png") for i in indices]
            combined = np.concatenate(imgs, axis=1)
            name = "_".join(str(i) for i in indices)
            imageio.imwrite(DEBUG_IMAGE_DIR / f"stack_{name}.png", combined)
    png_counter += 1


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        c, h, w = envs.single_observation_space.shape
        self.input_channels = c * STACK_SIZE

        self.n_actions = envs.single_action_space.nvec[0]

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(self.input_channels, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, h, w)
            n_flatten = self.network(dummy_input).shape[1]

        self.fc = nn.Sequential(
            layer_init(nn.Linear(n_flatten, 512)),
            nn.ReLU(),
        )

        # CHANGE 2: Actor now outputs a single vector of size 'n'
        self.actor = layer_init(nn.Linear(512, self.n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def get_value(self, x):
        return self.critic(self.fc(self.network(x / 255.0)))

    def get_action_and_value(self, x, action=None):
        hidden = self.fc(self.network(x / 255.0))
        logits = self.actor(hidden)

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


class TrainingMonitor:
    def __init__(self, num_envs=1):
        self.num_envs = num_envs
        self.start_time = time.time()

        # --- FPS Tracking State ---
        self.prev_step = 0
        self.prev_time = time.time()

        # --- Baseline Captures (System Idle State) ---
        gpus = GPUtil.getGPUs()
        self.gpu_total_mem = gpus[0].memoryTotal / 1024 if gpus else 0
        self.ram_total_gb = psutil.virtual_memory().total / (1024 ** 3)

        self.baseline_ram = psutil.virtual_memory().used / (1024 ** 3)
        self.baseline_vram = gpus[0].memoryUsed / 1024 if gpus else 0

        # --- Trackers for Peaks ---
        self.max_temp = 0
        self.max_ram = 0
        self.max_vram = 0
        self.max_cpu = 0

        # --- Disk/Time Tracking ---
        self.last_disk_usage = psutil.disk_io_counters()
        self.last_timestamp = time.time()

        psutil.cpu_percent(interval=None, percpu=True)

    def _get_dynamic_layout(self):
        ram_str_example = f"{self.ram_total_gb:.1f}GB / {self.ram_total_gb:.1f}GB"
        col_1 = max(32, len(ram_str_example) + 8)
        col_2 = 15
        line_width = col_1 + col_2 + 25
        return col_1, col_2, line_width

    def update(self, global_step):
        """Calculates FPS automatically and prints the dashboard."""
        now = time.time()

        # --- Internal FPS Calculation ---
        steps_delta = global_step - self.prev_step
        time_delta = now - self.prev_time
        fps = steps_delta / time_delta if time_delta > 0 else 0
        self.prev_step = global_step
        self.prev_time = now

        # --- Fetch Stats ---
        gpus = GPUtil.getGPUs()
        gpu = gpus[0] if gpus else None
        ram = psutil.virtual_memory()

        curr_cpu_pct = psutil.cpu_percent(interval=None)
        per_cpu_pct = psutil.cpu_percent(interval=None, percpu=True)
        cpu_freq_obj = psutil.cpu_freq()
        cpu_freq = cpu_freq_obj.current / 1000 if cpu_freq_obj else 0.0

        # --- Disk Logic ---
        dt = max(now - self.last_timestamp, 0.001)
        curr_disk_usage = psutil.disk_io_counters()
        if curr_disk_usage:
            r_speed = (curr_disk_usage.read_bytes - self.last_disk_usage.read_bytes) / (1024 ** 2) / dt
            w_speed = (curr_disk_usage.write_bytes - self.last_disk_usage.write_bytes) / (1024 ** 2) / dt
            self.last_disk_usage = curr_disk_usage
        else:
            r_speed = w_speed = 0.0
        self.last_timestamp = now

        # --- Memory Logic ---
        curr_ram_gb = ram.used / (1024 ** 3)
        curr_vram = gpu.memoryUsed / 1024 if gpu else 0

        # --- Update Peaks ---
        self.max_cpu = max(self.max_cpu, curr_cpu_pct)
        self.max_ram = max(self.max_ram, curr_ram_gb)
        self.max_vram = max(self.max_vram, curr_vram)
        self.max_temp = max(self.max_temp, gpu.temperature if gpu else 0)

        # --- UI Rendering ---
        col_1, col_2, line_width = self._get_dynamic_layout()

        print("\n" + "=" * line_width)
        print(f"STEP: {global_step:_} | FPS: {fps:.1f} | RUNTIME: {(now - self.start_time) / 60:.1f}m")
        print("-" * line_width)

        if gpu:
            vram_pct = (curr_vram / self.gpu_total_mem) * 100 if self.gpu_total_mem > 0 else 0
            print(f"TEMP:  {f'{gpu.temperature:.1f}°C':<{col_1}} | {f'--':^{col_2}} | Peak: {self.max_temp:.1f}°C")
            print(f"VRAM:  {f'{curr_vram:.2f}GB / {self.gpu_total_mem:.1f}GB':<{col_1}} | {f'{vram_pct:.0f}%':^{col_2}} | Peak: {self.max_vram:.2f}GB")
        else:
            print(f"GPU:   {'No GPU detected':<{col_1}}")

        print(f"RAM:   {f'{curr_ram_gb:.1f}GB / {self.ram_total_gb:.1f}GB':<{col_1}} | {f'{ram.percent:.0f}%':^{col_2}} | Peak: {self.max_ram:.1f}GB")
        print(f"CPU:   {f'{curr_cpu_pct:.1f}% @ {cpu_freq:.1f}GHz':<{col_1}} | {f'--':^{col_2}} | Peak: {self.max_cpu:.1f}%")

        core_map = "".join(["[!]" if p > 90 else "[~]" if p > 50 else "[.]" for p in per_cpu_pct])
        print(f"CORES: {core_map:<{col_1 + col_2 + 3}} | Threads: {len(per_cpu_pct)}")
        print(f"DISK:  {f'Read: {r_speed:.1f}MB/s | Write: {w_speed:.1f}MB/s':<{col_1}}")
        print("=" * line_width + "\n")


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

    print("\n -- Hardware stats  --")
    print(" -- if starting from the godot editor the stats count in the running game but for executables the first table does not count them in since the skript boots them up latter  --")
    monitor = TrainingMonitor(num_envs=args.num_envs)
    monitor.update(global_step=0)  # prints baseline system stats at startup

    resumed_wandb_run_id = None
    if args.checkpoint:
        _ckpt_peek = torch.load(args.checkpoint, map_location="cpu")
        resumed_wandb_run_id = _ckpt_peek.get("wandb_run_id", None)
        del _ckpt_peek

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            name=run_name,
            id=resumed_wandb_run_id,
            resume="allow" if resumed_wandb_run_id else None,
            config={**vars(args)},
            save_code=True,
            settings=wandb.Settings(console="wrap")
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    experiment_rewards = None
    envs = ImprovedGodotEnv(env_path=args.env_path, show_window=True, n_parallel=args.num_envs) # speedup=10 -> using this in here overwrites the speedup thats set in the sync node
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    start_update = 1
    if args.checkpoint:
        print(f"Resuming from checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        agent.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_step = ckpt["global_step"]
        start_update = ckpt["update"] + 1
        print(f"Resumed at global_step={global_step:_}, starting from update={start_update}")
        if args.track:
            append_resume_step(global_step)
    else:
        global_step = 0

    c, h, w = envs.single_observation_space.shape
    num_updates = args.total_timesteps // args.batch_size
    state_stack = np.zeros((args.num_envs, c * STACK_SIZE, h, w), dtype=np.float32)

    obs = torch.zeros((args.num_steps, args.num_envs, c * STACK_SIZE, h, w)).to(device)
    # CHANGE 5: Action buffer shape is now automatically correct for Discrete (scalar per env)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    next_obs, _ = envs.reset(seed=args.seed)

    for i in range(args.num_envs):
        frame = hex_to_ndarray(next_obs[i], (c, h, w))
        for s in range(STACK_SIZE):
            state_stack[i, s * c: (s + 1) * c, :, :] = frame

    next_obs_tensor = torch.Tensor(state_stack).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    moving_avg_window_size_wandb = 200
    moving_avg_window_size_cmd = 50
    scores = []  # [[episode_reward, global_step], ...]
    scores_count = np.zeros(args.num_envs)  # accumulates reward mid-episode per env
    episodes = 0
    last_steps = 0
    last_time = time.time()
    busy_lost_won_map = {"lost": 0, "won": 1}
    won_lost_list = [0] # avoid empty list problems
    ingame_score_list = [0]
    include_curriculum_learning = False
    curriculum_lvl2_at_step = 40_000_000 # the moving_avg_gamescore would prbably be a better trigger
    curriculum_lvl2_sent_counter = 0

    if args.track:
        if include_curriculum_learning:
            wandb.config.update({"#curriculum_lvl2_at_step": f"{curriculum_lvl2_at_step:_}"})

    update = start_update  # safety fallback in case of interrupt before loop starts
    try:
        for update in range(start_update, num_updates + 1):
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                optimizer.param_groups[0]["lr"] = frac * args.learning_rate

            for step in range(0, args.num_steps):
                global_step += 1 * args.num_envs
                obs[step] = next_obs_tensor
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs_tensor)
                    values[step] = value.flatten()
                actions[step] = action.unsqueeze(-1)
                logprobs[step] = logprob

                raw_obs, reward, terminated, truncated, infos = envs.step(
                    action.cpu().numpy().reshape(-1, 1)  # reshape (num_envs,) -> (num_envs, 1)
                )
                if args.track:
                    if not experiment_rewards:
                        experiment_rewards = infos[0].get("experiment_rewards") # infos[0] just takes it from the first env
                        print("experiment_rewards")
                        print(experiment_rewards)
                        if experiment_rewards:
                            wandb.config.update({"#experiment_rewards": str(experiment_rewards)})

                #print("reward, terminated, truncated, infos ")
                #print(reward, terminated, truncated, infos )

                try: # note: if the done comes along with lost/won then it could be moved there for a bit of optimization
                    for blw in infos:
                        if "in_game_score" in blw:
                            ingame_score_list.append(blw["in_game_score"])
                        if blw.get("busy_lost_won") in ["lost", "won"]:
                            won_lost_list.append(busy_lost_won_map[blw["busy_lost_won"]])

                except Exception as e: # -> in case its not coming from godot get_info()
                    # print(f"busy_lost_won Error: {e}")
                    pass # --- WARNING --- the try/except could mess with the performance -> comment out if unused

                done = np.logical_or(terminated, truncated)

                for i in range(args.num_envs):
                    new_frame = hex_to_ndarray(raw_obs[i], (c, h, w))
                    if done[i]:
                        for s in range(STACK_SIZE):
                            state_stack[i, s * c: (s + 1) * c, :, :] = new_frame
                    else:
                        state_stack[i, :-c, :, :] = state_stack[i, c:, :, :]
                        state_stack[i, -c:, :, :] = new_frame

                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs_tensor = torch.Tensor(state_stack).to(device)
                next_done = torch.tensor(done).to(device).float()

                if include_curriculum_learning:
                    # if global_step >= 2_00 and curriculum_lvl2_sent_counter <= 10: # fast debugging
                    if global_step >= curriculum_lvl2_at_step and curriculum_lvl2_sent_counter <= 10:
                        print("=============Trigger Curriculum Learning Logic===========")
                        print("--- PHASE 2: Increasing Difficulty ---")
                        envs.set_curriculum(2, # one int of your choice
                                            [1, 2, 3], # a list of numbers of your choice
                                            ["placeholder_1", "placeholder_2", "placeholder_3"]) # a list of strings of your choice
                        curriculum_lvl2_sent_counter += 1

                if global_step % 42_000 == 0 and len(scores) > 0:
                    avg_score = np.mean([entry[0] for entry in scores[-moving_avg_window_size_cmd:]])
                    window_games = won_lost_list[-moving_avg_window_size_cmd:]
                    moving_win_loss_ratio = sum(window_games) / len(window_games)
                    avg_ingame_score = np.mean(ingame_score_list[-moving_avg_window_size_cmd:])
                    print(
                        f"PPO avg score {avg_score:.2f} | "
                        f"avg ingame points {avg_ingame_score:.0f} | "
                        f"games {episodes} |"
                        f"total_timesteps {global_step:_} | "  # underscore instead of comma
                        f"Win/Loss ratio {moving_win_loss_ratio:.2f} | "
                        f"fps {(global_step - last_steps) / (time.time() - last_time):.2f} | ",
                        flush=True
                    )
                    last_steps = global_step
                    last_time = time.time()

                    monitor.update(global_step=global_step)

                for i in range(args.num_envs):
                    scores_count[i] += reward[i]
                    if done[i]:
                        scores.append([scores_count[i], global_step])
                        episodes += 1
                        recent_scores = [entry[0] for entry in scores[-moving_avg_window_size_wandb:]]
                        current_avg = float(np.mean(recent_scores))
                        # print(
                        #     f"global_step={global_step}, episodic_return={scores_count[i]:.2f}, moving_avg={current_avg:.2f}")
                        window_games = won_lost_list[-moving_avg_window_size_wandb:]
                        moving_win_loss_ratio = sum(window_games) / moving_avg_window_size_wandb
                        avg_ingame_score = np.mean(ingame_score_list[-moving_avg_window_size_wandb:])

                        if args.track:
                            wandb.log({
                                "charts/moving_avg_reward": round(current_avg, 3),
                                "charts/moving_win_loss_ratio": moving_win_loss_ratio,
                                "charts/moving_avg_game_score": avg_ingame_score,
                            }, step=global_step)
                        # # uncomment if you want to debug scores -> if 1 invader is 0.1 and a life loss is -1 it should be easy to calculate if the scores make sense
                        # print("all scores so far: ")
                        # print(scores)
                        # print("current avg so far: ", current_avg)
                        scores_count[i] = 0.0

            with torch.no_grad():
                next_value = agent.get_value(next_obs_tensor).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            b_obs = obs.reshape((-1, c * STACK_SIZE, h, w))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            mb_inds = np.arange(args.batch_size)
            for epoch in range(args.update_epochs):
                np.random.shuffle(mb_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    micro_indices = mb_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[micro_indices],
                        b_actions[micro_indices].long().squeeze(-1)
                    )

                    logratio = newlogprob - b_logprobs[micro_indices]
                    ratio = logratio.exp()

                    mb_advantages = b_advantages[micro_indices]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    v_loss = 0.5 * ((newvalue.view(-1) - b_returns[micro_indices]) ** 2).mean()
                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()


            if args.track:
                wandb.log({
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                }, step=global_step)

            if args.track and global_step % 250_000 < args.batch_size:
                torch.save({
                    "global_step": global_step,
                    "update": update,
                    "model_state_dict": agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "wandb_run_id": wandb.run.id,
                }, "checkpoint.pth")
                print(f"Checkpoint saved at step {global_step:_}")

        # clean completion — save full resumable checkpoint
        if args.track:
            torch.save({
                "global_step": global_step,
                "update": update,
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "wandb_run_id": wandb.run.id,
            }, "checkpoint.pth")
            wandb.save("checkpoint.pth", policy="now")
            print("Training complete. Final checkpoint saved and uploaded to wandb.")
        else:
            print("Training complete. (track=False, checkpoint not saved)")

        # also save weights-only copy for inference
        final_model_path = f"runs/{run_name}/model.pth"
        torch.save(agent.state_dict(), final_model_path)
        print(f"Weights-only model saved to {final_model_path}")

        # onnx will be taken care of correctly in future courses
        if args.save_onnx:
            agent.eval()
            dummy_input = torch.randn(1, c * STACK_SIZE, h, w).to(device)
            onnx_path = f"runs/{run_name}/model.onnx"
            torch.onnx.export(agent, dummy_input, onnx_path, input_names=['input'], output_names=['output'])
            print(f"ONNX model saved to {onnx_path}")

    except KeyboardInterrupt:
        if args.track:
            print("Interrupted! Saving and uploading checkpoint...")
            torch.save({
                "global_step": global_step,
                "update": update,
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "wandb_run_id": wandb.run.id,
            }, "checkpoint.pth")
            wandb.save("checkpoint.pth", policy="now")
            print("Checkpoint saved and uploaded to wandb.")
        else:
            print("Interrupted! (track=False, checkpoint not saved)")

    finally:
        envs.close()
        if args.track:
            wandb.finish()