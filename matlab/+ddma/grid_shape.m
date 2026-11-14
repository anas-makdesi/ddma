function shape = grid_shape(edges)
%GRID_SHAPE Number of cells along each coordinate (row vector).
shape = cellfun(@numel, edges(:)') - 1;
end
