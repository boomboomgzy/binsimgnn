from utils import init_nodevector
import os
import sys





if __name__=='__main__':

    prop_threads=50
    debug=False
    small_dataset=True
    root=r'/home/ouyangchao/binsimgnn'
    save_dir=os.path.join(root,'dataset')

    if debug:
        homoG_save_dir=os.path.join(save_dir,'test_homoG')
        ir_programl_dir=os.path.join(save_dir,'test_ir_programl')
        vocab_dir=os.path.join(root,'test_vocab')

    else:
        if small_dataset:
            homoG_save_dir=os.path.join(save_dir,'binkit_small_homoG')
            ir_programl_dir=os.path.join(save_dir,'binkit_small_ir_programl')
            vocab_dir=os.path.join(root,'binkit_small_vocab')
 
        else:         
            print('please choose a dataset')   
            sys.exit(1)

    corpus_model_path=os.path.join(vocab_dir,'ir_corpus.model')
    corpus_vec_path=os.path.join(vocab_dir,'ir_corpus.vector')
    vector_log_file=os.path.join(vocab_dir,'vetor_log.txt')


    init_nodevector(ir_programl_dir,homoG_save_dir,corpus_model_path,corpus_vec_path,prop_threads,vector_log_file)
    #初始化后要重新构建数据集
