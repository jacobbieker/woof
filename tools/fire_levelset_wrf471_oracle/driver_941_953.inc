          s = 1./(ir*jr)
          do j=jts,jte
            do i=its,ite
                ! DME heat fluxes contribution for the case wiythout feedback
                grnhfx_fu(i,j)=grnhfx(i,j)*s
                grnqfx_fu(i,j)=grnqfx(i,j)*s
                ! scale surface fluxes to get the averages
                avg_fuel_frac(i,j)=avg_fuel_frac(i,j)*s
                grnhfx(i,j)=fire_atm_feedback*grnhfx(i,j)*s
                grnqfx(i,j)=fire_atm_feedback*grnqfx(i,j)*s
                ! we do not have canopy fluxes yet...
                canhfx(i,j)=0
                canqfx(i,j)=0
