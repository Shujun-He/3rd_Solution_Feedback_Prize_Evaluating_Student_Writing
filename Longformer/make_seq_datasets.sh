for i in {0..7};do
  nohup python make_seq_datasets_clean_v2.py --fold ${i}> ${i}_seqds.out&
done
