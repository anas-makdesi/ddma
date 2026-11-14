function mask = cells_inside(edges, lo, hi)
%CELLS_INSIDE Logical column vector of the cells contained in the box [lo, hi].
[clo, chi] = ddma.cell_bounds(edges, ddma.all_subs(ddma.grid_shape(edges)));
mask = all(clo >= lo(:)' & chi <= hi(:)', 2);
end
