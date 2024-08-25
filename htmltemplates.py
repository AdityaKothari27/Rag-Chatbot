css = '''
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Text:ital@0;1&display=swap');
.chat-message {
    padding: 1.5rem; 
    border-radius: 0.5rem; 
    margin-bottom: 1rem; 
    display: flex; 
    font-family: "DM Serif Text", serif;
    font-weight: 400;
    font-style: normal;
}
.chat-message.user {
    background-color: #000000;
    border: 1px solid #ffffff;
}
.chat-message.bot {
    background-color: #000000;
    border: 1px solid #ffffff;
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar .avatar-text {
  color: #fff;
  font-weight: bold;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <div class="avatar-text">Bot:</div>
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <div class="avatar-text">User:</div>
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''



# css = '''
# <style>
# .chat-message {
#     padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
# }
# .chat-message.user {
#     background-color: #2b313e
# }
# .chat-message.bot {
#     background-color: #475063
# }
# .chat-message .avatar {
#   width: 20%;
#   font-color: #fff;
# }
# .chat-message .avatar img {
#   max-width: 78px;
#   max-height: 78px;
#   border-radius: 50%;
#   object-fit: cover;
#   font-color: #fff;
# }
# .chat-message .message {
#   width: 80%;
#   padding: 0 1.5rem;
#   color: #fff;
# }
# '''

# bot_template = '''
# <div class="chat-message bot">
#     <div class="avatar">
#         <img src="https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.freepik.com%2Ffree-photos-vectors%2Fbot&psig=AOvVaw2kc8QTDQdTjb7POkOyVPju&ust=1720520774419000&source=images&cd=vfe&opi=89978449&ved=0CBEQjRxqFwoTCKCPtMCdl4cDFQAAAAAdAAAAABAE">
#     </div>
#     <div class="message">{{MSG}}</div>
# </div>
# '''

# user_template = '''
# <div class="chat-message user">
#     <div class="avatar">
#         User:
#     </div>    
#     <div class="message">{{MSG}}</div>
# </div>
# '''