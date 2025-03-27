import os

import imageio
from moviepy.editor import VideoFileClip


def export_gif(config_name, frames, gif_save_path, timestamp_str):
    # 文件夹
    gif_folder = os.path.join(gif_save_path, f"{timestamp_str}_{config_name}")
    os.makedirs(gif_folder, exist_ok=True)

    # 1. gif生成
    gif_path = os.path.join(
        gif_folder,
        f"multiwalker_{config_name}.gif",
    )
    imageio.mimwrite(
        gif_path,
        frames,
        duration=10,
    )

    # 3. 视频生成
    clip = VideoFileClip(gif_path)
    clip.write_videofile(
        os.path.join(
            gif_folder,
            f"multiwalker_{config_name}.mp4",
        ),
        codec="libx264",
        logger=None,
    )
