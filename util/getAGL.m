function agl = getAGL( sitename )
% getAGL returns the altitude above ground layer for a site specified by
% the parameter sitename
% Examples:
%   altitude = getAGL( 'F6' );

    agl = 37; % per default!
    
    if( strcmpi( sitename, 'F6' ) )
       agl = 34; 
    elseif( strcmpi( sitename, 'F5' ) )
        agl = 32;
    elseif( strcmpi( sitename, 'F0' ) )
        agl = 45;
	elseif( strcmpi( sitename, 'F1' ) )
        agl = 28;
    elseif( strcmpi( sitename, 'F2' ) )
        agl = 32;
    elseif( strcmpi( sitename, 'F3' ) )
        agl = 41;
    elseif( strcmpi( sitename, 'F4' ) )
        agl = 40;
    elseif( strcmpi( sitename, 'T2' ) )
        agl = 32;
    elseif( strcmpi( sitename, 'T3' ) ) % chosen better value not the best one
        agl = 33;
    elseif( strcmpi( sitename, 'T4' ) )
        agl = 35;
    elseif( strcmpi( sitename, 'T6' ) ) % chosen better value not the best one
        agl = 32;
    elseif( strcmpi( sitename, 'F8' ) )
        agl = 24;
    elseif( strcmpi( sitename, 'F9' ) ) 
        agl = 26;
    elseif( strcmpi( sitename, 'F10' ) ) 
        agl = 14;
    elseif( strcmpi( sitename, 'F11' ) ) 
        agl = 25;
    end
end
