import numpy as np
import tensorflow as tf
import math
import sys
import matplotlib.pyplot as plt

np.random.seed(1)
tf.set_random_seed(1)


class Generator(object):
    def __init__(self,M,N,**kwargs):
        self.M = M
        self.N = N
        vars(self).update(kwargs)
        self.x_ = tf.placeholder( tf.complex64,(M,1),name='x' )
        self.y_ = tf.placeholder( tf.complex64,(N,1),name='y' )
        self.H_ = tf.placeholder( tf.complex64,(N,M),name='H')
        self.SNR_ = tf.placeholder( tf.float32, name='SNR')


class TFGenerator(Generator):
    def __init__(self,**kwargs):
        Generator.__init__(self,**kwargs)
    def __call__(self,sess):#TFGenerate(sess)
        'generates y,x pair for training'
        return sess.run( ( self.xgen_,self.Hgen_ ) )


def random_channel(M=30,N=60,L=10000):
    H_real = tf.random_normal( (N,M),stddev=math.sqrt( 1/2/N) )
    H_imag = tf.random_normal( (N,M),stddev=math.sqrt( 1/2/N) )
    Hgen_ = tf.complex(H_real, H_imag)

    prob = TFGenerator(M=M, N=N)
    prob.name = 'Gaussian, random H'
    prob.Htype = 0
    x_real = tf.sign(tf.random_uniform((M,L),minval=-1,maxval=1,dtype=tf.float32))
    x_imag = tf.zeros((M,L))
    xgen_ = tf.complex(x_real, x_imag)
    SNR = prob.SNR_
    noise_var = M/N * tf.pow(10., -SNR / 10.)
    noise_real = tf.random_normal( (N,L),stddev=tf.sqrt( noise_var/2) )
    noise_imag = tf.random_normal( (N,L),stddev=tf.sqrt( noise_var/2) )
    noise_ = tf.complex(noise_real, noise_imag)
    ygen_ = tf.matmul( prob.H_, prob.x_) + noise_
    prob.xgen_ = xgen_
    prob.ygen_ = ygen_
    prob.Hgen_ = Hgen_
    prob.L = L
    return prob



def build_Lcg(prob,T=3, type = 0):
    # type 0 scalar alpha,beta
    # type 1 vector alpha，beta
    layers=[]
    M = prob.M
    N = prob.N
    L = prob.L
    H = prob.H_
    Hreal = tf.real(H)
    Himag = tf.imag(H)
    H_Real_left = tf.concat([Hreal, Himag],0)
    H_Real_right = tf.concat([-1*Himag, Hreal], 0)
    H_Real = tf.concat([H_Real_left, H_Real_right], 1)
    y_Real = tf.concat([tf.real(prob.y_), tf.imag(prob.y_)], 0)
    SNR = prob.SNR_
    noise_var = M / N * tf.pow(10., -SNR / 10.)
    A = tf.matmul(tf.transpose(H_Real),H_Real) + noise_var * tf.eye(2*M)
    b = tf.matmul(tf.transpose(H_Real), y_Real)
    d_ = b
    r_ = d_
    if type==0:
      alpha_ = tf.Variable(0, dtype=tf.float32, name='alpha_0')
    else:
      init_zero = np.zeros((2*M,1))
      alpha_ = tf.Variable(init_zero, dtype=tf.float32, name='alpha_0')
    xhat_ = alpha_ * d_
    layers.append(('LcgNet TYPE={0} T=0'.format(type), xhat_, (alpha_,),()))
    for t in range(1,T+1):
        rlast_ = r_
        r_ = r_ - alpha_ * tf.matmul(A, d_)
        if type==0:
          beta_ = tf.Variable(0, dtype=tf.float32, name='beta_' + str(t))
        else:
          beta_ = tf.Variable(init_zero, dtype=tf.float32, name='beta_' + str(t))
        d_ = r_ + beta_ * d_
        if type==0:
          alpha_ = tf.Variable(0, dtype=tf.float32, name='alpha_' + str(t))
        else:
          alpha_ = tf.Variable(init_zero, dtype=tf.float32, name='alpha_' + str(t))
        xhat_ = xhat_ + alpha_ * d_
        layers.append(('LcgNet type={0} T={1}'.format(type,t), xhat_, (alpha_, beta_),()))
    return layers


def save_trainable_vars(sess,filename,**kwargs):
    save={}
    for v in tf.trainable_variables():
        save[str(v.name)] = sess.run(v)
    save.update(kwargs)
    np.savez(filename,**save)

def load_trainable_vars(sess,filename):
    other={}
    try:
        tv=dict([ (str(v.name),v) for v in tf.trainable_variables() ])
        for k,d in np.load(filename).items():
            if k in tv:
                print('restoring ' + k)
                sess.run(tf.assign( tv[k], d) )
            else:
                other[k] = d
    except IOError:
        pass
    return other


def setup_training(layer_info,prob, trinit=1e-3,refinements=(.5,.1,.01),final_refine=None):
    nmse_=[]
    assert np.array(refinements).min()>0,'all refinements must be in (0,1]'
    assert np.array(refinements).max()<=1,'all refinements must be in (0,1]'

    x_Real_ = tf.concat([tf.real(prob.x_), tf.imag(prob.x_)], 0)
    nmse_denom_ = tf.nn.l2_loss(x_Real_)
    tr_ = tf.Variable(trinit,name='tr',trainable=False)
    training_stages=[]
    for name,xhat_,var_list,TanhVarlist in layer_info:
        loss_  = tf.nn.l2_loss( xhat_ - x_Real_)
        nmse_  = loss_ / nmse_denom_
        if var_list is not None:
            train_ = tf.train.AdamOptimizer(tr_).minimize(loss_, var_list=var_list)
            training_stages.append((name, xhat_, loss_, nmse_, train_, var_list,()))
        for fm in refinements:
            train2_ = tf.train.AdamOptimizer(tr_*fm).minimize(loss_)
            training_stages.append( (name+' train rate=' + str(fm) ,xhat_,loss_,nmse_,train2_,(),()) )
    if final_refine:
        train2_ = tf.train.AdamOptimizer(tr_*final_refine).minimize(loss_)
        training_stages.append( (name+' final refine ' + str(final_refine) ,xhat_,loss_,nmse_,train2_,(),()) )
    return training_stages


def do_training(training_stages,prob,savefile1,savefile2, ivl=10,maxit=100,better_wait=500,SNR=30.,ValNum=100):

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    M = prob.M
    N = prob.N
    L = prob.L
    noise_var = M / N * math.pow(10., -SNR / 10.)

    HvalSet = []
    xvalSet = []
    yvalSet = []
    for i in range(ValNum):
      xval = np.sign(np.random.uniform(-1, 1, (M, L))) + 1j * np.zeros((M, L))
      Hval = np.random.normal(size=(N, M), scale=1.0 / math.sqrt(N * 2)).astype(np.float32) + 1j * np.random.normal(
        size=(N, M), scale=1.0 / math.sqrt(N * 2)).astype(np.float32)
      noise = np.random.normal(size=(N, L), scale=math.sqrt(noise_var / 2)).astype(np.float32) + 1j * np.random.normal(size=(N, L), scale=math.sqrt(noise_var / 2)).astype(np.float32)
      yval = np.matmul(Hval, xval) + noise
      HvalSet.append(Hval)
      xvalSet.append(xval)
      yvalSet.append(yval)
    state = load_trainable_vars(sess,savefile1)

    done=state.get('done',[])
    log=str(state.get('log',''))

    for name,xhat_,loss_,nmse_,train_,var_list,TanhVarlist in training_stages:
        name_SNR = name + ' with SNR={SNR: .6f}'.format(SNR=SNR)
        if name_SNR in done:
            print('Already did ' + name_SNR + '. Skipping.')
            continue
        if len(var_list):
            describe_var_list = 'extending ' + ','.join([v.name for v in var_list])
        else:
            describe_var_list = 'fine tuning all ' + ','.join([v.name for v in tf.trainable_variables() ])
        print(name_SNR + ' ' + describe_var_list )
        nmse_history=[]
        for i in range(maxit+1):
            if i%ivl == 0:
                NMSE = []
                for j in range(ValNum):
                  xval=xvalSet[j]
                  yval=yvalSet[j]
                  Hval=HvalSet[j]
                  nmse_temp = sess.run(nmse_,feed_dict={prob.y_:yval,prob.x_:xval,prob.H_:Hval,prob.SNR_:SNR})
                  NMSE = np.append(NMSE, nmse_temp)
                nmse = np.mean(NMSE)
                if np.isnan(nmse):
                    raise RuntimeError('nmse is NaN')
                nmse_history = np.append(nmse_history,nmse)
                nmse_dB = 10*np.log10(nmse)
                nmsebest_dB = 10*np.log10(nmse_history.min())
                sys.stdout.write('\ri={i:<6d} nmse={nmse:.6f} dB (best={best:.6f})'.format(i=i,nmse=nmse_dB,best=nmsebest_dB))
                sys.stdout.flush()
                if i%(100*ivl) == 0:
                    print('')
                    age_of_best = len(nmse_history) - nmse_history.argmin()-1
                    if age_of_best*ivl > better_wait:
                        break
            x,H = prob(sess)
            y = sess.run(prob.ygen_,feed_dict={prob.x_:x,prob.H_:H, prob.SNR_:SNR} )
            sess.run(train_,feed_dict={prob.y_:y,prob.x_:x,prob.H_:H, prob.SNR_:SNR} )

        done = np.append(done,name_SNR)

        log =  log+'\n{name} nmse={nmse:.6f} dB in {i} iterations'.format(name=name_SNR,nmse=nmse_dB,i=i)

        state['done'] = done
        state['log'] = log
        save_trainable_vars(sess,savefile2,**state)
    return sess

def do_test(training_stages,prob,savefile,TestNum=10000,SeeVar=1):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    state = load_trainable_vars(sess, savefile)
    M = prob.M
    N = prob.N
    L = prob.L
    if (SeeVar==1):
        for v in tf.trainable_variables():
            print(v.name)
            print(sess.run(v))
    Num = TestNum
    BER_test = []
    name, xhat_, loss_, nmse_, train_, var_list,TanhVarlist = training_stages[-1]
    for i in range(0,6):
        ErrNum = 0
        SNR = i*2
        for j in range(1, Num+1):
            xTest = np.sign(np.random.uniform(-1,1,(M,L))) + 1j* np.zeros((M,L))
            xHard=  np.real(xTest)
            H_Test = np.random.normal(size=(N, M), scale=1.0 / math.sqrt(N*2)).astype(np.float32) + 1j * np.random.normal(size=(N, M), scale=1.0 / math.sqrt(N*2)).astype(np.float32)
            noise_var = M / N * math.pow(10., -SNR / 10.)
            noise_Test = np.random.normal(size=(N, L), scale=math.sqrt(noise_var / 2)).astype(np.float32) + 1j * np.random.normal(size=(N, L), scale=math.sqrt(noise_var / 2)).astype(np.float32)
            yTest = np.matmul(H_Test, xTest) + noise_Test
            NMSEtemp = sess.run(nmse_,feed_dict={prob.y_:yTest,prob.x_:xTest,prob.H_:H_Test, prob.SNR_:SNR} )
            #start = datetime.datetime.now()
            xhat = sess.run(xhat_,feed_dict={prob.y_:yTest,prob.x_:xTest,prob.H_:H_Test, prob.SNR_:SNR} )
            #end = datetime.datetime.now()
            #print(end - start) #calculate time

            xhat_Hard = np.sign(xhat[0:M])
            ErrTemp = (np.array(np.where((xHard - xhat_Hard)!=0)))[0].size
            ErrNum = ErrNum + ErrTemp
        ber = ErrNum / Num / M
        BER_test = np.append(BER_test, ber)
        print('Test SNR={snr: .6f} BER={ber:.10f}'.format(snr=SNR,ber=ber))
    plt.plot(np.log10(BER_test))
    return sess

# System Model, BPSK
prob = random_channel(M=32,N=64,L=1)
#Construct Network;
layers = build_Lcg(prob,T=15, type=1)
#Set up training stage
training_stages= setup_training(layers,prob,trinit=1e-3,refinements=(0.5,) )
#training
#for snr in (30,):
  #sess = do_training(training_stages, prob, 'LcgNetV_3264.npz', 'LcgNetV_3264.npz', maxit=30000,SNR=snr, ValNum=100)
# testing
sess = do_test(training_stages,prob,'LcgNetV_3264.npz',TestNum=100000)