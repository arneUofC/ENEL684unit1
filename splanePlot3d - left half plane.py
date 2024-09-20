import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import control as ct
import enel441_utilities as eu

S = ct.tf([1, 4, 6, 4], [1, 4, 7, 7])
print(S.zeros())
print(S.poles())

#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


num = np.array([1, 4, 6, 4])
den = np.array([1, 4, 7, 7])

# Make data.
alpha = np.arange(-5,0,0.05)
beta = np.arange(-5,5,0.05)

S_mag = np.zeros((alpha.shape[0], beta.shape[0]))
ii = 0
for aa in alpha:
    jj = 0
    for bb in beta:
        s = complex(aa,bb)
        S_mag[ii,jj] = np.abs((s**3 + 4*s**2 + 6*s + 4)/(s**3 + 4*s**2 + 7*s + 7))
        if S_mag[ii,jj] > 100:
            S_mag[ii,jj] = 100
        jj+=1
    ii+=1
        


Alpha, Beta = np.meshgrid(beta, alpha)

print(alpha.shape)
print(beta.shape)
print(S_mag.shape)
print(Alpha.shape)
print(Beta.shape)

fig=plt.figure()

ax = fig.add_subplot(2,2,1)
eu.enel441_s_plane_plot(num, den, fig=fig, ax=ax)

ax = fig.add_subplot(2,2,3,projection='3d')
# Plot the surface.
surf = ax.plot_surface(Alpha, Beta, S_mag, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(0, 5)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')
ax.set_ylabel('Real')
ax.set_xlabel('Imaginary')
ax.set_title('Magnitude of $G(s)$ in S-Plane')
ax.plot(beta,np.zeros(beta.shape[0]),S_mag[-1,:],'k',linewidth=3)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

ax = fig.add_subplot(2,2,2)
ax.plot(beta,S_mag[-1,:])
ax.set_xlabel('Imaginary')
ax.set_ylabel('Magnitude')
ax.set_title('Magnitude of $G(s)$ along line $s=jB$')

n_half = int(beta.shape[0]/2)
ax=fig.add_subplot(2,2,4)
ax.semilogx(beta[n_half:-1],10*np.log10(np.abs(S_mag[-1,n_half:-1])))
ax.set_xlabel('Frequency (rad/s)')
ax.set_title('Magnitude of $G(j\omega)$')
ax.set_ylabel('Magnitude (dB)')
plt.show()

