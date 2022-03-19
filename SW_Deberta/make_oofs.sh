for i in {0..7};do
  nohup python make_oofs.py --fold ${i} --gpu_id ${i}> ${i}_seqds.out&
done
