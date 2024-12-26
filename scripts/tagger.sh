python modules/stable/tag_images_by_wd14_tagger.py \
--onnx \
--repo_id SmilingWolf/wd-swinv2-tagger-v3 \
--batch_size 4 \
--remove_underscore \
--undesired_tags "" \
--recursive \
--use_rating_tags_as_last_tag \
--character_tags_first \
--character_tag_expand \
--always_first_tags "chinese_painting"  \
train/datasets/birme-1024x1024
