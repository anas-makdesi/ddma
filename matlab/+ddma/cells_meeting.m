function mask = cells_meeting(edges, lo, hi)
%CELLS_MEETING Logical column vector of the cells whose closure meets the box [lo, hi].
%   Conservative (closed cells): the right choice for obstacles.
%   Same as the Python SymbolicAbstraction.cells_meeting.
[clo, chi] = ddma.cell_bounds(edges, ddma.all_subs(ddma.grid_shape(edges)));
mask = all(chi >= lo(:)' & clo <= hi(:)', 2);
end
