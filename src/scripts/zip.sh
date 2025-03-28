#!/bin/bash

# 创建一个新的zip文件
zip_file="selected_folders.zip"
rm -f $zip_file
zip -r $zip_file /dev/null 2>/dev/null

# 遍历mappo目录及其子目录
shopt -s globstar
for dir in ./results/pettingzoo_mw/multiwalker/happo/**/; do
  if [[ "$dir" == *"0327-2334" ]]; then
    # zip -r $zip_file "$dir"
    echo "$dir"
  fi
done
