
# coding: utf-8

# Flask.app code

# from flask import Flask,request, render_template, request, url_for
# import pickle
# import numpy as np
# 
# app = Flask(__name__)
# app.config["DEBUG"] = True
# 
# comments= []
# 
# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "GET":
#         return render_template("main_page.html", comments=comments)
# 
#     comments.append(request.form["contents"])
#     return redirect(url_for('index'))
# 
# #Prepare the feature vector for prediction
#         pkl_file = open('cat', 'rb')
#         index_dict = pickle.load(pkl_file)
#         new_vector = np.zeros(len(index_dict))
# 
# if __name__ == '__main__':
#     app.run()
# 

# Template for the home page

# <html>
#     <head>
#         <meta charset="utf-8">
# <meta http-equiv="X-UA-Compatible" content="IE=edge">
# <meta name="viewport" content="width=device-width, initial-scale=1">
# 
# <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/css/bootstrap.min.css" integrity="sha512-dTfge/zgoMYpP7QbHy4gWMEGsbsdZeCXz7irItjcC3sPUFtf0kuFbDz/ixG7ArTxmDjLXDmezHubeNikyKGVyQ==" crossorigin="anonymous">
#         <title>Mohi's Bitcoin Prediction Model </title>
#     </head>
# 
#     <body>
#         <nav class="navbar navbar-inverse">
#     <div class="container">
#         <div class="navbar-header">
#             <button type="button" class="navbar-toggle collapsed" data-toggle="collapse" data-target="#navbar" aria-expanded="false" aria-controls="navbar">
#                 <span class="sr-only">Toggle navigation</span>
#                 <span class="icon-bar"></span>
#                 <span class="icon-bar"></span>
#                 <span class="icon-bar"></span>
#             </button>
#             <a class="navbar-brand" href="#">Mohi's Bitcoin Prediction Model</a>
#         </div>
#     </div>
# </nav>
# 
# 
# <div class="container">
# 
#         <div class="row">
#             Welcome to my Bitcoin Prediction Model
#         </div>
# 
#         <div class="row">
#             You need to enter today's closing price for the NASDAQ and KOSDAQ in USD
#         </div>
# 
#         <div class="row">
#             Then you need to use Google Trends to enter this week's "impression" of the word "bitcoin" (scale of 1 to 100)
#         </div>
# 
#         <div class="row">
#             <form action="." method="POST">
#                 <textarea name="contents" placeholder="NASDAQ $$"></textarea>
#                 <input type="submit" class="btn btn-success" value="ENTER NASDAQ Price">
# 
#             </form>
#         </div>
#         <div class="row">
#             <form action="." method="POST">
#                 <textarea name="contents" placeholder="KOSDAQ $$"></textarea>
#                 <input type="submit" class="btn btn-success" value="ENTER KOSDAQ Price">
# 
#             </form>
#         </div>
#         <div class="row">
#             <form action="." method="POST">
#                 <textarea name="contents" placeholder="Impressions Count (1-100)"></textarea>
#                 <input type="submit" class="btn btn-success" value="ENTER Impressions Count">
# 
#             </form>
# 
#         </div>
# 
# </div><!-- /.container -->
#     </body>
# </html>
