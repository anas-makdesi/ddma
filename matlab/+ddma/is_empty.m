function e = is_empty(S)
%IS_EMPTY Cells where lower > upper in some output (never for consistent data).
e = any(S.lower > S.upper, 2);
end
