#!/bin/bash
# python src/fgvc/train.py experiment=plant_traits description="vitb_pclef_blk_4_bld_train_tokens" model.model.train_blocks=4 data.batch_size=32

# python src/fgvc/train.py experiment=plant_traits description="vitb_pclef_blk_5_bld_train_tokens" model.model.train_blocks=5 data.batch_size=32

# python src/fgvc/train.py experiment=plant_traits description="vitb_pclef_blk_6_bld_train_tokens" model.model.train_blocks=6 data.batch_size=16

# python src/fgvc/train.py experiment=plant_traits description="vitb_pclef_blk_7_bld_train_tokens" model.model.train_blocks=7 data.batch_size=16

# python src/fgvc/train.py experiment=plant_traits description="vitb_pclef_blk_8_bld_train_tokens" model.model.train_blocks=8 data.batch_size=16

# python src/fgvc/train.py experiment=plant_traits description="v4 clf sft" model.reg_traits=False model.clf_traits=False model.soft_clf_traits=True model.bld_traits=False val_on="val/soft_clf_r2"
python src/fgvc/train.py experiment=plant_traits description="v4 clf" model.reg_traits=False model.clf_traits=True model.soft_clf_traits=False model.bld_traits=False val_on="val/clf_r2"

python src/fgvc/train.py experiment=plant_traits description="v4_all_v2_all_data" model.reg_traits=True model.clf_traits=True model.soft_clf_traits=True model.bld_traits=True val_on="val/blend_r2" data.batch_size=32

python src/fgvc/train.py experiment=plant_traits description="v4_all_v2_vitl_all_data" model.reg_traits=True model.clf_traits=True model.soft_clf_traits=True model.bld_traits=True val_on="val/blend_r2" data.batch_size=16

