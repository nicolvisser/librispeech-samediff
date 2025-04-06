# samediff-ed --subset dev-clean \
# --units-dir /mnt/wsl/data/units_duped/wavlm-large/layer-24/k-512/LibriSpeech/dev-clean \
# --unit-rate 50 \
# --log-dir /mnt/wsl/nvme/code/librispeech-samediff/tmp \
# --run-name "units-wavlm-large-layer-24-k-512"

# for k in 64 128 192 256 320 384 448 512; do
#     samediff-ed --subset dev-clean \
#     --units-dir "/mnt/wsl/data/units_duped/wavlm-large/layer-24/k-512-naive-cut-${k}/LibriSpeech/dev-clean" \
#     --unit-rate 50 \
#     --log-dir /mnt/wsl/nvme/code/librispeech-samediff/tmp \
#     --run-name "units-wavlm-large-layer-24-k-512-naive-cut-${k}"
# done

# for k in 32 64 96 128 185; do
#     samediff-ed --subset dev-clean \
#     --units-dir "/mnt/wsl/data/units_duped/wavlm-large/layer-24/k-512-mashlm-cut-${k}/LibriSpeech/dev-clean" \
#     --unit-rate 50 \
#     --log-dir /mnt/wsl/nvme/code/librispeech-samediff/tmp \
#     --run-name "units-wavlm-large-layer-24-k-512-mashlm-cut-${k}"
# done


samediff-ed --subset dev-clean \
--units-dir "/mnt/wsl/data/units_duped/wavlm-large/layer-24/k-512-mashlm-cut-32/LibriSpeech/dev-clean" \
--unit-rate 50 \
--log-dir /mnt/wsl/nvme/code/librispeech-samediff/tmp \
--run-name "units-wavlm-large-layer-24-k-512-mashlm-cut-32-sanity"
