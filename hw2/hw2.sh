# CartPole-v0 small batch
python pg.py CartPole-v0 -n 100 -b 1000 -e 3 -dna --exp_name sb_no_rtg_dna
python pg.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg -dna --exp_name sb_rtg_dna
python pg.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg --exp_name sb_rtg_na

# CartPole-v0 large batch
python pg.py CartPole-v0 -n 100 -b 5000 -e 3 -dna --exp_name lb_no_rtg_dna
python pg.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg -dna --exp_name lb_rtg_dna
python pg.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg --exp_name lb_rtg_na

# InvertedPendulum-v2 stddev
python pg.py InvertedPendulum-v2 -ep 1000 --discount 0.99 -n 100 -e 3 -l 2 -s 64 -b 1000 -lr 0.005 -rtg --exp_name hc_b1000_r0.005_stddev0.1
python pg_stddev.py InvertedPendulum-v2 -ep 1000 --discount 0.99 -n 100 -e 3 -l 2 -s 64 -b 1000 -lr 0.005 -rtg --exp_name hc_b1000_r0.005_stddevlearned
python pg_stddev_simple.py InvertedPendulum-v2 -ep 1000 --discount 0.99 -n 100 -e 3 -l 2 -s 64 -b 1000 -lr 0.005 -rtg --exp_name hc_b1000_r0.005_stddevlearnednet

# InvertedPendulum-v2 discount
python pg.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b 1000 -lr 0.005 -rtg --exp_name hc_b1000_r0.005_disc0.9
python pg.py InvertedPendulum-v2 -ep 1000 --discount 0.99 -n 100 -e 3 -l 2 -s 64 -b 1000 -lr 0.005 -rtg --exp_name hc_b1000_r0.005_disc0.99
python pg.py InvertedPendulum-v2 -ep 1000 --discount 1. -n 100 -e 3 -l 2 -s 64 -b 1000 -lr 0.005 -rtg --exp_name hc_b1000_r0.005_disc1

# InvertedPendulum-v2 discount
python pg_stddev.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b 1000 -lr 0.005 -rtg --exp_name hc_b1000_r0.005_other_disc0.9
python pg_stddev.py InvertedPendulum-v2 -ep 1000 --discount 0.99 -n 100 -e 3 -l 2 -s 64 -b 1000 -lr 0.005 -rtg --exp_name hc_b1000_r0.005_other_disc0.99
python pg_stddev.py InvertedPendulum-v2 -ep 1000 --discount 1. -n 100 -e 3 -l 2 -s 64 -b 1000 -lr 0.005 -rtg --exp_name hc_b1000_r0.005_other_disc1
