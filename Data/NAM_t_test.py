import time
import numpy as np
import os
import netCDF4
from sklearn.decomposition import PCA
pca = PCA()
import proplot as pplt
repertory = "/Users/mariaccia/Documents/Etude_Scenarios2/Compute_NAM/Data_ERA5/"
NETCDF_files = os.listdir(repertory)
Mensual_mean = np.load('/Users/mariaccia/Documents/Etude_Scenarios2/Compute_NAM/Results/Daily_Zonal_Mean_Z.npy')

#%%
""" 
    On calcule l'EOF des anomalies de géopotentiel moyennées zonallement sur chaque niveau 
    de pression (Start here)
"""
ano_z = np.load('/Users/mariaccia/Documents/Etude_Scenarios2/Compute_NAM/Results/Ano_Z_daily.npy')
EOFp = np.ndarray((37, 71)) # 37 niveaux de pression et 71 latitudes
ano_z_p = np.zeros((37, 71, 25915)) # 25915 dates de 71 latitudes
for p in range(37):
    for i in range(71): # on boucle sur tous les hivers
        ano_z_p[p, :, i*365:(i+1)*365] = ano_z[i, :, p, :].T
    EOF = pca.fit_transform(ano_z_p[p, :, :])
    print("sklearn var:\n", pca.explained_variance_ratio_[0])
    EOFp[p, :] = EOF[:, 0]
print("done \n")
#%%
""" 
    On calcule les indices de NAM en projetant les anomalies sur l'EOF 
"""
t0 = time.time()
Indices = np.zeros((71, 365, 37))
W = np.cos(np.arange(90, 19, -1) * np.pi/180)

for y in range(71):
    for d in range(365):
        for p in range(37):
            Indices[y, d, p] = np.dot(ano_z[y, d, p, :], W * EOFp[p, :])/np.dot(EOFp[p, :].T, W * EOFp[p, :])

""" Normalisation des indices entre -1 et 1 par niveau de pression """

Indices = Indices.reshape(71 * 365, 37)
for i in range(37):
    Indices[:, i] = (Indices[:, i] - np.mean(Indices[:, i])) / np.std(Indices[:, i])
    # Indices[:, i] = (((Indices[:, i] - min(Indices[:, i]))*2)/(max(Indices[:, i]) - min(Indices[:, i]))) - 1
Indices = Indices.reshape(71, 365, 37)
print('done')
print("done \n")
print(time.time() - t0)
#%%
""" On construit les 70 hivers du 1er Nov au 1er Mai """
import datetime
t0 = datetime.datetime.now()
times = np.array([np.datetime64("2019-01-01") + np.timedelta64(n, 'D') for n in range(0,365)])

idx1 = np.where(np.datetime64("2019-11-01") <= times)[0]
idx2 = np.where((np.datetime64("2019-05-01") >= times))[0]
Indices_Winters = np.ndarray((70, len(idx1)+len(idx2), 37))
for i in range(len(Indices_Winters)):
    Indices_Winters[i, 0:len(idx1), :] = Indices[i, idx1, :]
    Indices_Winters[i, len(idx1):, :] = Indices[i+1, idx2, :]

print("done")
print(datetime.datetime.now() - t0)


#%%

""" On fait un t test sur """
import scipy.stats as stats

t0 = datetime.datetime.now()
modes = [np.array([0, 2, 4, 9, 17, 19, 20, 26, 34, 47, 51, 52, 53, 55, 61, 62, 68]),
         np.array([6, 7, 12, 22, 28, 30, 32, 36, 38, 39, 40, 44, 57, 58, 59, 66, 67]),
         np.array([1, 15, 18, 29, 37, 48, 50]),
         np.array([5, 8, 10, 13, 23, 24, 25, 35, 42, 45, 49, 60, 63, 64, 65]),
         np.array([11, 14, 16, 46, 69])]

# perform one sample t-test
Tab_signi = np.ndarray((5, 182, 37))
Tab_signi[:, :, :] = False
for m in range(5):
    for i in range(182):
        for j in range(37):
            tab_test = Indices_Winters[modes[m], i, j]
            t_statistic, p_value = stats.ttest_1samp(a=tab_test, popmean=0)
            # t_statistic, p_value = stats.ttest_1samp(a=tab_test, popmean=np.mean(Indices_Winters[:, i, j], axis=0))
            if p_value <= 0.05:
                Tab_signi[m, i, j] = True

print('done')
print(datetime.datetime.now() - t0)



#%%
""" On plot les moyennes des indices de chaque mode """


t0 = datetime.datetime.now()
titles = ["January Mode", "February Mode", "Double Mode", "Dynamical Final Warming Mode", "Radiative Final Warming Mode"]#, "Early Warming Mode"]
mode_early = np.array([1952, 1958, 1966, 1968, 1974, 1976, 1979, 1987, 1996, 2000, 2009, 2016])
years1 = np.arange(1950, 2021)
idx_early = np.intersect1d(years1, mode_early, return_indices=True)[1]

modes = [np.array([0, 2, 4, 9, 17, 19, 20, 26, 34, 47, 51, 52, 53, 55, 61, 62, 68]),
         np.array([6, 7, 12, 22, 28, 30, 32, 36, 38, 39, 40, 44, 57, 58, 59, 66, 67]),
         np.array([1, 15, 18, 29, 37, 48, 50]),
         np.array([5, 8, 10, 13, 23, 24, 25, 35, 42, 45, 49, 60, 63, 64, 65]),
         np.array([11, 14, 16, 46, 69]),
         idx_early]

levels = np.array([ 1,    2,    3,    5,    7,   10,   20,   30,   50,
                    70,  100,  125,  150,  175,  200,  225,  250,  300,
                    350,  400,  450,  500,  550,  600,  650,  700,  750,
                    775,  800,  825,  850,  875,  900,  925,  950,  975,
                    1000])

times1 = np.array([np.datetime64("2019-01-01") + np.timedelta64(n, 'D') for n in range(0,365)])
times2 = np.array([np.datetime64("2018-01-01") + np.timedelta64(n, 'D') for n in range(0,365)])
timesf = np.concatenate([times2[idx1], times1[idx2]])

for i, title in enumerate(titles):
    fig = pplt.figure(grid=False, refaspect=2)
    ax = fig.subplot(111)
    # fig, ax = pplt.subplots(nrows=1, share=False, yreverse=True, grid=False, space=0, refaspect=2)
    Id = np.transpose(np.mean(Indices_Winters[modes[i], :, :], axis=0))
    m = ax.contourf(timesf, levels, Id,
                    cmap='Div', extend='both', levels=pplt.arange(-2, 2, 0.25),
                    cmap_kw={'cut': -0.05})
    ax.plot(timesf, np.ones((len(timesf)))*250, c='k', lw=0.25, ls='-')
    ix, iy = np.where(Tab_signi[i, :, :])
    new_i, new_j = [], []
    for j in range(len(ix)):
        new_i.append(timesf[ix[j]])
        new_j.append(levels[iy[j]])
    # ax.scatter(new_i, new_j, color='k', marker='o', markersize=0.1)
    ax.contour(timesf, levels, np.transpose(Tab_signi[i]), nozero=True, color='gray8',
               labels=False, lw=0.5, locator=1)#, labels_kw={'weight': ''})
    # ax.colorbar(m, loc='r')
    # ax.text(np.datetime64('2018-10-08'), 5, '(hPa)')
    ax.format(yscale='log', title=title,
              xlim=(np.datetime64('2018-11-01'), np.datetime64('2019-05-01')),
              # xlim=(np.datetime64('1999-01-01'), np.datetime64('1999-03-31T12')),
              xformatter='%b', xminorlocator='month', xrotation=0,
              ylabel='Pressure (hPa)', yreverse=True, grid=False)
    fig.colorbar(m, loc='r', length=1, width=0.1, ticks=0.5)
    fig.savefig("/Users/mariaccia/Documents/Etude_Scenarios2/Compute_NAM/NAM_t_test/"+title+".png", dpi=1000)
    pplt.close('all')
print("done")

#%%
""" On plot les 3 scénarios perturbés """
pplt.rc.update({'fontname': 'Source Sans Pro', 'fontsize': 10})
t0 = datetime.datetime.now()
titles = ["January Mode", "February Mode", "Double Mode"]
years1 = np.arange(1950, 2021)

modes = [np.array([0, 2, 4, 9, 17, 19, 20, 26, 34, 47, 51, 52, 53, 55, 61, 62, 68]),
         np.array([6, 7, 12, 22, 28, 30, 32, 36, 38, 39, 40, 44, 57, 58, 59, 66, 67]),
         np.array([1, 15, 18, 29, 37, 48, 50]),]

levels = np.array([ 1,    2,    3,    5,    7,   10,   20,   30,   50,
                    70,  100,  125,  150,  175,  200,  225,  250,  300,
                    350,  400,  450,  500,  550,  600,  650,  700,  750,
                    775,  800,  825,  850,  875,  900,  925,  950,  975,
                    1000])

times1 = np.array([np.datetime64("2019-01-01") + np.timedelta64(n, 'D') for n in range(0,365)])
times2 = np.array([np.datetime64("2018-01-01") + np.timedelta64(n, 'D') for n in range(0,365)])
timesf = np.concatenate([times2[idx1], times1[idx2]])

fig, axs = pplt.subplots(ncols=1, nrows=3, sharey=False, sharex=False, refaspect=2)
for i, title in enumerate(titles):
    ax = axs[i]
    # fig, ax = pplt.subplots(nrows=1, share=False, yreverse=True, grid=False, space=0, refaspect=2)
    Id = np.transpose(np.mean(Indices_Winters[modes[i], :, :], axis=0))
    m = ax.contourf(timesf, levels, Id,#/np.max(np.abs(Id)),
                    cmap='Div', extend='both', levels=pplt.arange(-2, 2, 0.25),
                    cmap_kw={'cut': -0.05})
    ax.plot(timesf, np.ones((len(timesf)))*250, c='k', lw=0.25, ls='--')
    ix, iy = np.where(Tab_signi[i, :, :])
    new_i, new_j = [], []
    for j in range(len(ix)):
        new_i.append(timesf[ix[j]])
        new_j.append(levels[iy[j]])
    # ax.scatter(new_i, new_j, color='k', marker='o', markersize=0.1)
    ax.contour(timesf, levels, np.transpose(Tab_signi[i]), nozero=True, color='gray8',
               labels=False, lw=0.5, locator=1)#, labels_kw={'weight': ''})
    # ax.colorbar(m, loc='r')
    # ax.text(np.datetime64('2018-10-08'), 5, '(hPa)')
    ax.format(yscale='log', title=title,
              xlim=(np.datetime64('2018-11-01'), np.datetime64('2019-05-01')),
              # xlim=(np.datetime64('1999-01-01'), np.datetime64('1999-03-31T12')),
              xformatter='%b', xminorlocator='month', xrotation=0,
              ylabel='Pressure (hPa)', yreverse=True, grid=False)
fig.colorbar(m, loc='r', length=1, width=0.1, ticks=0.5)

fig.format(abc='a.')
fig.savefig("/Users/mariaccia/Documents/Etude_Scenarios2/Compute_NAM/NAM_t_test/3_modes.png", dpi=1000)
pplt.close('all')
print("done")
#%%
""" On plot les 2 scénarios non perturbés """

t0 = datetime.datetime.now()
titles = ["Dynamical Final Warming Mode", "Radiative Final Warming Mode"]
years1 = np.arange(1950, 2021)

modes = [np.array([5, 8, 10, 13, 23, 24, 25, 35, 42, 45, 49, 60, 63, 64, 65]),
         np.array([11, 14, 16, 46, 69]),]

levels = np.array([ 1,    2,    3,    5,    7,   10,   20,   30,   50,
                    70,  100,  125,  150,  175,  200,  225,  250,  300,
                    350,  400,  450,  500,  550,  600,  650,  700,  750,
                    775,  800,  825,  850,  875,  900,  925,  950,  975,
                    1000])

times1 = np.array([np.datetime64("2019-01-01") + np.timedelta64(n, 'D') for n in range(0,365)])
times2 = np.array([np.datetime64("2018-01-01") + np.timedelta64(n, 'D') for n in range(0,365)])
timesf = np.concatenate([times2[idx1], times1[idx2]])

fig, axs = pplt.subplots(ncols=1, nrows=2, sharey=False, sharex=False, refaspect=2)
for i, title in enumerate(titles):
    ax = axs[i]
    # fig, ax = pplt.subplots(nrows=1, share=False, yreverse=True, grid=False, space=0, refaspect=2)
    Id = np.transpose(np.mean(Indices_Winters[modes[i], :, :], axis=0))
    m = ax.contourf(timesf, levels, Id,
                    cmap='Div', extend='both', levels=pplt.arange(-2, 2, 0.25),
                    cmap_kw={'cut': -0.05})
    ax.plot(timesf, np.ones((len(timesf)))*250, c='k', lw=0.25, ls='--')
    ix, iy = np.where(Tab_signi[i, :, :])
    new_i, new_j = [], []
    for j in range(len(ix)):
        new_i.append(timesf[ix[j]])
        new_j.append(levels[iy[j]])
    # ax.scatter(new_i, new_j, color='k', marker='o', markersize=0.1)
    ax.contour(timesf, levels, np.transpose(Tab_signi[i+3]), nozero=True, color='gray8',
               labels=False, lw=0.5, locator=1)#, labels_kw={'weight': ''})
    # ax.colorbar(m, loc='r')
    # ax.text(np.datetime64('2018-10-08'), 5, '(hPa)')
    ax.format(yscale='log', title=title,
              xlim=(np.datetime64('2018-11-01'), np.datetime64('2019-05-01')),
              # xlim=(np.datetime64('1999-01-01'), np.datetime64('1999-03-31T12')),
              xformatter='%b', xminorlocator='month', xrotation=0,
              ylabel='Pressure (hPa)', yreverse=True, grid=False)
fig.colorbar(m, loc='r', length=1, width=0.1, ticks=0.5)

fig.format(abc='a.')
fig.savefig("/Users/mariaccia/Documents/Etude_Scenarios2/Compute_NAM/NAM_t_test/2_modes.png", dpi=1000)
pplt.close('all')
print("done")