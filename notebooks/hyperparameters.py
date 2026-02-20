from configs.constants import *


#### MU SNN parameters
mu_params_leaky = {'S1':{TAU_SYN_COL: 0.01,  # set to 0 when tau_syn is learnt and not fixed
                    TAU_MEM_COL: 0.08, # set to zero when tau_mem is learnt and not fixed. Otherwise, it is fixed
                    TAU_SYN_INIT_COL: 0, # set to 0 when tau_syn is learnt and not fixed
                    TAU_MEM_INIT_COL: 0,
                    TAU1_MEM_FILT_COL: 0.08,# this filter was not used in the post process. It is the default value
                    TAU2_MEM_FILT_COL: 0.0,
                    EPOCHS_COL: 100,
                    },
                    'S2':{TAU_SYN_COL: 0.01,  # set to 0 when tau_syn is learnt and not fixed
                    TAU_MEM_COL: 0.08, # set to zero when tau_mem is learnt and not fixed. Otherwise, it is fixed
                    TAU_SYN_INIT_COL: 0, # set to 0 when tau_syn is learnt and not fixed
                    TAU_MEM_INIT_COL: 0,
                    TAU1_MEM_FILT_COL: 0.08,# this filter was not used in the post process. It is the default value
                    TAU2_MEM_FILT_COL: 0.0,
                    EPOCHS_COL: 100,
                    }
        }

mu_params_spiking = {'S1':{TAU_SYN_COL: 0.01,  # 0  when tau_syn is learnt and not fixed
                      TAU_MEM_COL: 0,  # this is zero when tau_mem is learnt and not fixed
                      TAU_SYN_INIT_COL: 0,
                      TAU_MEM_INIT_COL: 0.03,
                      # these two consecutive time constants are used when two filters are used. Otherwise, they are set to 0
                      TAU1_MEM_FILT_COL: 0.05,
                      TAU2_MEM_FILT_COL: 0.08,
                      EPOCHS_COL: 100,
                      },
                      'S2':{TAU_SYN_COL: 0.01,  # 0  when tau_syn is learnt and not fixed
                      TAU_MEM_COL: 0,  # this is zero when tau_mem is learnt and not fixed
                      TAU_SYN_INIT_COL: 0,
                      TAU_MEM_INIT_COL: 0.03,
                      # these two consecutive time constants are used when two filters are used. Otherwise, they are set to 0
                      TAU1_MEM_FILT_COL: 0.05,
                      TAU2_MEM_FILT_COL: 0.08,
                      EPOCHS_COL: 100,
                      }
}


#### Encoding SNN parameters
enc_params_leaky = {'S1': {TAU_SYN_COL: 0.010,
                           TAU_SYN_INIT_COL: 0,
                            TAU_MEM_COL: 0.08,
                            TAU_MEM_INIT_COL: 0,
                            TAU1_MEM_FILT_COL: 0.2,
                            TAU2_MEM_FILT_COL: 0.0,
                            EPOCHS_COL: 100,
                            ENC_TAU_MEM1_COL: 0.02,
                            ENC_TAU_MEM2_COL: 0.02,
                            ENC_TAU_MEM3_COL: 0.02,
                            ENC_SPK_THRESH1_COL: 0.1,
                            ENC_SPK_THRESH2_COL:0.1,
                            ENC_SPK_THRESH3_COL: 0.1,
                            ENC_ALIF_SCALE_COL: 0.1,
                            ENC_THRESHOLD_TAU_COL: 0.1,
                            USE_ALIF_COL: True,

                        # 
                        #     ENC_SPK_THRESH2_COL: 0.4,
                        #     ENC_SPK_THRESH3_COL: 0.2,
                            },
                    'S2': {TAU_SYN_COL: 0.010,
                            TAU_SYN_INIT_COL: 0,

                            TAU_MEM_COL: 0.08,
                            TAU_MEM_INIT_COL: 0,
                            TAU1_MEM_FILT_COL: 0.2,
                            TAU2_MEM_FILT_COL: 0.0,
                            EPOCHS_COL: 100,
                            ENC_TAU_MEM1_COL: 0.02,
                            ENC_TAU_MEM2_COL: 0.02,
                            ENC_TAU_MEM3_COL: 0.02,
                            ENC_SPK_THRESH1_COL: 0.1,
                            ENC_SPK_THRESH2_COL:0.1,
                            ENC_SPK_THRESH3_COL: 0.1,
                            ENC_ALIF_SCALE_COL: 0.1,
                            ENC_THRESHOLD_TAU_COL: 0.1,
                            USE_ALIF_COL: True,
                        #     ENC_SPK_THRESH1_COL: 0.06,
                        #     ENC_SPK_THRESH2_COL: 0.06,
                        #     ENC_SPK_THRESH3_COL: 0.06,
                            }
                    }