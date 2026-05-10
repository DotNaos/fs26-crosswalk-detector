# Local Remote Dashboard

This setup keeps the UI on `localhost` and lets the local service submit Slurm jobs to the school GPU server.

## What runs where

- Your browser opens the dashboard on `http://127.0.0.1:8787`
- The local Bun service stores config, starts `tmux` sessions, uploads job JSON, watches Slurm, and downloads results
- The GPU server only runs the classification job and returns a compact result JSON

## One-time setup

1. Create the local state folder:

```bash
mkdir -p .local/remote-controller
```

2. Copy the example config:

```bash
cp remote-controller.config.example.json .local/remote-controller/config.json
```

3. Set the SSH password as an environment variable:

```bash
export CROSSWALK_REMOTE_PASSWORD='your-password-here'
```

4. Start the local container:

```bash
docker compose -f docker-compose.local.yml up --build
```

## Daily use

1. Open [http://127.0.0.1:8787](http://127.0.0.1:8787)
2. Open a scene and zoom into the tile field
3. In the `Remote` panel:
   - check the host / username / repo path
   - click `Connect`
   - click `Run Current Circle`
4. Watch the job state and log output in the same panel
5. When the job finishes, the result is imported back into the map automatically

## Notes

- `tmux` is required because the local controller uses one detached session per remote job
- The password is intentionally not stored in JSON or committed files
- If you use SSH keys instead, leave `CROSSWALK_REMOTE_PASSWORD` unset and the controller will try plain `ssh`
