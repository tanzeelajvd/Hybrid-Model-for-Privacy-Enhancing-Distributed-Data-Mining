import random
from decimal import *

from phe import paillier

getcontext().prec = 50


# This code was working with only three input points now every step is generalized
# to work with any number of points i.e. any list can be passed and it will return sp
# here p is denoting the point

def rss_rmd_sum(p_input):
    p_modified = list(p_input)  # taking copy

    # these two parameters are not used in the code..
    public_key, private_key = paillier.generate_paillier_keypair()

    num_points = len(p_modified)
    # print('......Step 1: Modify input parameters..................')
    for i in range(num_points):
        alpha_pi = random.uniform(-0.5, 0.5)
        p_modified[i] = p_modified[i] + alpha_pi

    # print('...........Step 2: Generating the parameters..................')
    r = list()
    f = list()
    g = list()
    R = list()
    val = 16 ** 30
    for i in range(num_points):
        r.append(random.uniform(3000, 5000))
        f.append(list())
        g.append(list())
        R.append(list())
        for j in range(num_points):
            f[i].append(random.uniform(20000000, 50000000))
            g[i].append(random.uniform(20000000, 50000000))
            random_hex = '0x' + '%031x' % random.randrange(val)
            R[i].append(int(random_hex, 16))

    # print("...................STEP3..................")
    pr = list()
    for i in range(num_points):
        pr.append(list())
        for j in range(num_points):
            if i == j:
                pr[i].append(0)
                continue
            pr[i].append(round(Decimal(r[i]) * Decimal(f[i][j])))
            public_key.encrypt(pr[i][j])

    # print("..................STEP4......................")
    gp = list()
    for i in range(num_points):
        gp.append(list())
        for j in range(num_points):
            if i == j:
                gp[i].append(0)
                continue
            gp[i].append(round(Decimal(g[i][j]) * Decimal(p_modified[i])))

    rep = list()
    for i in range(num_points):
        rep.append(list())
        for j in range(num_points):
            if i == j:
                rep[i].append(0)
                continue
            rep[i].append(pr[i][j] * gp[j][i] + R[j][i])

    # print("..................STEP5......................")
    share = list()
    share_dup = list()
    for i in range(num_points):
        share.append(list())
        share_dup.append(list())
        for j in range(num_points):
            if i == j:
                share[i].append(0)
                share_dup[i].append(0)
                continue
            share[i].append(Decimal(rep[i][j]) / (Decimal(f[i][j]) * Decimal(g[j][i])))
            share_dup[i].append(Decimal(R[i][j]) / (Decimal(f[j][i]) * Decimal(g[i][j])))

    # print("..................STEP6......................")
    phi = []
    for i in range(num_points):
        phi.append(random.uniform(-0.5, 0.5))

    rfp = []
    for i in range(num_points):
        rfp.append((Decimal(r[i]) + Decimal(phi[i])) * Decimal(p_modified[i]))

    s = []
    for i in range(num_points):
        share_sum = sum(map(Decimal, share[i]))
        share_dup_sum = sum(map(Decimal, share_dup[i]))
        final_val = share_sum - share_dup_sum + Decimal(rfp[i])
        s.append(final_val)

    sp = sum(s)
    return sp


if __name__ == '__main__':
    print(rss_rmd_sum([10, 10, 10, 10]) / rss_rmd_sum([1, 1, 1, 1]))
