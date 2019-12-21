<html>
	<head>
		<title>RepoDash - Create visual metrics of your Github Issues</title>
	</head>
	<body>
		<header>
		<ul>
                        <li><a href="/about">Requirements</a></li>
	        	<li><a href="/about">User Guide</a></li>
        		<li><a href="/cv">Technical Documentation</a></li>
		</ul>
      		</header>
		<div class="container">
    		<div class="blurb">
        		<h1>Welcome to RepoDash</h1>
				<p>Do you maintain a project codebase on Github? 
           Do you spend an awful lot of your time managing your project's issues list? 
           Would you like to be able to take a step back and answer key questions, such as:</p>
        <ul>
        <li>Do you have it under control?</li>
        <li>Can you resolve issues faster than they are created?</li> 
        <li>On average, how long do issues remain on the list before they are resolved?</li>
        </ul>
        <p>To answer questions like these, I have created a tool that uses the Github API to generate a 
        data visualisation that displays some key metrics of the Github issues list of any Github project. 
        This could be just what you need to provide your boss with a clear and unambiguous exective summary 
        of your team's progress, a kind of support and maintenance state-of-the-union. 
        Or perhaps you have a personal project, the maintenance of which you'd like to keep on top of.</p>
        <p>This code displays matplotlib metrics by default (a popular open science plotting library for python). 
        To point it at your project, simply provide your account name and repository name on the command line.
        It couldn't be simpler.</p>
        <p>NOTE: This has only been tested with open source (public) projects right now, but I have plans to make 
        sure it works with private ones as well... watch this space!</p>
    		</div><!-- /.blurb -->
		</div><!-- /.container -->
		<footer>
    		<ul>
        		<li><a href="mailto:laurence.molloy@gmail.com">email</a></li>
        		<li><a href="https://github.com/laurencemolloy">github.com/laurencemolloy</a></li>
			</ul>
		</footer>
	</body>
</html>
