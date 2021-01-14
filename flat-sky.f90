!Flat-sky formula from Michel Aguena, Sij, for a cylindrical window function
!with a radius theta<=10deg, for bins of redshifts directly (approximating that stuff varies slowly with redshift)
!S(iz,jz) = int dlnkperp kperp^2 4 J1(kperp theta r1)/(kperp theta r1) J1(kperp theta r2)/(kperp theta r2)
!               * int dkpar j0(kpar dr1/2) j0(kpar dr2/2) cos(kpar(r1-r2)) P(k)/(2pi^2)
!with k = sqrt(kpar^2 + kperp^2) ; r1 is the comoving distance to the center of iz, and dr1 the distance width
!r2 and dr2 are the corresponding for redshift bin jz

!Some rewriting
!Given than j0(x)=sinx/x and sina*sinb*cosc = 1/4 * [cos(a-b+c)+cos(a-b-c)-cos(a+b-c)-cos(a+b+c)]
! int dkpar .. = int dkpar P(k)/(2pi^2)*(kpar*dr1/2*kpar*dr2/2)^(-1)*1/4*[cos(kpar*(dr1/2-dr2/2+r1-r2))+cos(kpar*(dr1/2-dr2/2-r1+r2))-cos(kpar*(dr1/2+dr2/2-r1+r2))-cos(kpar*(dr1/2+dr2/2+r1-r2))]
! ... = (dr1/2*dr2/2)^(-1) * 1/4 * [DCT_kpar(dr1/2-dr2/2+r1-r2) + DCT_kpar(dr1/2-dr2/2-r1+r2) - DCT_kpar(dr1/2+dr2/2-r1+r2)- DCT_kpar(dr1/2+dr2/2+r1-r2)]
! with DCT_kpar[x] = int dkpar P(k)/(2pi^2) kpar^(-2) cos(kpar*x) 
!calling Dr12=abs(r1-r2), eps12=abs(dr1/2-dr2/2), eta12=dr1/2+dr2/2, we have
! int dkpar .. = [DCT_kpar(Dr12+eps12)+DCT_kpar(Dr12-eps12)-DCT_kpar(Dr12-eta12)-DCT_kpar(Dr12+eta12)] / (dr1*dr2)

! Approximation with second derivative d2_DCT_kpar, if eps12, eta12 -> 0 :
! int dkpar .. = d2_DCT_kpar(Dr12)*[eps12^2 - eta12^2] / (dr1*dr2) = -d2_DCT_kpar(Dr12)
! which is equal to the integral for dr1=dr2=0 i.e. case of infinitesimally thin redshift bins
! corrections are of order O(dr1^2,dr2^2)

theta = sqrt(4./768.)
!kperp array
kmin_perp = SSC%kmin_sigma2
kmax_perp = SSC%kmax_sigma2
nk_perp   = SSC%nk_sigma2
Do ik=1,nk_perp
  lnkperp_arr(ik) = (log(kmax_perp) - log(kmin_perp))*(ik-1.)/(nk_perp-1.) + log(kmin_perp)
  kperp_arr(ik) = exp(lnkperp_arr(ik))
End Do
!FFT along kpar of kpar^(-2)*P(sqrt(kpar^2 + kperp^2)) for different kperp
kparm2Pk_2bfftd                = 0
wsave                          = 0
kparm2Pk_azero                 = 0
kparm2Pk_dct                   = 0
kparm2Pk_dst                   = 0
d2_kparm2Pk_dct                = 0
kparm2Pk_dct_cwzm_eachkperp    = 0
d2_kparm2Pk_dct_cwzm_eachkperp = 0
dr_fft                         = 0
kmin_par = SSC%kmin_sigma2
kmax_par = SSC%kmax_sigma2
!Note: nk_par = nk_fft
Call dzffti(nk_fft,wsave)
Do jk=1,(nk_fft/2-1)
  dr_fft(jk) = 2.d+0 * pi * jk / (kmax_par-kmin_par)  !Fourier wavevectors
End Do
dr_fft_wzm(1) = 0.d+0
dr_fft_wzm(2:nk_fft/2) = dr_fft
drmin    = minval(dr_fft)
drmax    = maxval(dr_fft)
Do ik=1,nk_perp
  kpe2 = kperp_arr(ik)**2
  Do jk=1,nk_fft
    !linear grid on kpar, as we need to use an fft
    kpar          = (kmax_par - kmin_par)*(jk-1.)/(nk_fft-1.) + kmin_par 
    ktot          = sqrt(kpe2 + kpar**2)
    kparm2Pk_2bfftd(jk) = P_dd_ln(ktot,0.0d+0)/(2.*pi**2)/kpar**2
  End Do
  Call dzfftf(nk_fft,kparm2Pk_2bfftd,kparm2Pk_azero,kparm2Pk_dct,kparm2Pk_dst,wsave)
  kparm2Pk_dct_cwzm(1)          = kparm2Pk_azero   !Include zero mode
  kparm2Pk_dct_cwzm(2:nk_fft/2) = kparm2Pk_dct/2.  !Because of factor 2 in definition of FFTPACK
  Call spline(dr_fft_wzm,kparm2Pk_dct_cwzm,(nk_fft/2),3.d30,3.d30,d2_kparm2Pk_dct_cwzm) !! computes second derivative
  kparm2Pk_dct_cwzm_eachkperp(:,ik)    = kparm2Pk_dct_cwzm
  d2_kparm2Pk_dct_cwzm_eachkperp(:,ik) = d2_kparm2Pk_dct_cwzm
End Do

!now do the integral over kperp (which depends on z1,z2)
Do j1 = 1,nzbins
  dr1   = abs(comov_dist(nz_perbin,j1)-comov_dist(1,j1))
  r1moy = (comov_dist(nz_perbin,j1)+comov_dist(1,j1))/2.
  growth1 = fast_DoD0(z_arr(nz_perbin/2,j1)) !Growth factor at the center of the bin
  Do j2 = 1,j1
    dr2   = abs(comov_dist(nz_perbin,j2)-comov_dist(1,j2))
    r2moy = (comov_dist(nz_perbin,j2)+comov_dist(1,j2))/2.
    growth2 = fast_DoD0(z_arr(nz_perbin/2,j2))
    Dr12  = abs(r1moy-r2moy)
    eps12 = abs(dr1/2.-dr2/2.)
    eta12 = dr1/2.+dr2/2.
    Dr_arr_4Sij(1) = Dr12 + eps12
    Dr_arr_4Sij(2) = abs(Dr12 - eps12)
    Dr_arr_4Sij(3) = Dr12 + eta12
    Dr_arr_4Sij(4) = abs(Dr12 - eta12)
    Do ik=1,nk_perp
      kperp     = kperp_arr(ik)
      x1        = kperp * theta * r1moy
      x2        = kperp * theta * r2moy
      truc1     = Bessel_J1(x1)/x1
      truc2     = Bessel_J1(x2)/x2
      kparm2Pk_dct_cwzm    = kparm2Pk_dct_cwzm_eachkperp(:,ik)
      d2_kparm2Pk_dct_cwzm = d2_kparm2Pk_dct_cwzm_eachkperp(:,ik)
      Do i=1,4
        if (Dr_arr_4Sij(i) < drmax) then
          Call splint(dr_fft_wzm,kparm2Pk_dct_cwzm,d2_kparm2Pk_dct_cwzm,(nk_fft/2-1),Dr_arr_4Sij(i),DCT_arr_4Sij(i))
        else
          DCT_arr_4Sij(i) = 0.0d+0  !Cannot interpolate reliably if outside the range
       End If
      End Do
      int_kpar = (DCT_arr_4Sij(1) + DCT_arr_4Sij(2) - DCT_arr_4Sij(3) - DCT_arr_4Sij(4)) / (dr1*dr2)
      integrand_kperp(ik) = kperp**2 * 4. * truc1 * truc2 * int_kpar
    End Do
    !Integrating and putting back the growth functions
    Sij(j1,j2) = integrate_vec(lnkperp_arr,integrand_kperp,iint) * growth1 * growth2
  End Do
End Do
