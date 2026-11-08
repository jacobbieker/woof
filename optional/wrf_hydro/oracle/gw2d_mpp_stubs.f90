! module_gw_gw2d.F90's several-process ADI pieces; one process never calls them.
subroutine sub_n_form()
  error stop 'sub_n_form: one process does not split a sweep'
end subroutine
subroutine parysolv1()
  error stop 'parysolv1: one process does not split a sweep'
end subroutine
subroutine parxsolv1()
  error stop 'parxsolv1: one process does not split a sweep'
end subroutine
