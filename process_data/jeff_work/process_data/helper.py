import html2text
import codecs
from bs4 import BeautifulSoup



html = """<body>
<div style="color: #b5b5b5;">##- Please type your reply above this line -##</div>
<p></p><div style="margin-top: 25px" data-version="2"><table width="100%" cellpadding="0" cellspacing="0" border="0" role="presentation">  <tr>    <td width="100%" style="padding: 15px 0; border-top: 1px dotted #c5c5c5;">      <table width="100%" cellpadding="0" cellspacing="0" border="0" style="table-layout:fixed;" role="presentation">        <tr>                    <td width="100%" style="padding: 0; margin: 0;" valign="top">            <p style="font-family: 'Lucida Grande','Lucida Sans Unicode','Lucida Sans',Verdana,Tahoma,sans-serif; font-size: 15px; line-height: 18px; margin-bottom: 0; margin-top: 0; padding: 0; color:#1b1d1e;">                                                                    <strong>Jonathan</strong> (Edison Mail)                                                            </p>            <p style="font-family: 'Lucida Grande','Lucida Sans Unicode','Lucida Sans',Verdana,Tahoma,sans-serif; font-size: 13px; line-height: 25px; margin-bottom: 15px; margin-top: 0; padding: 0; color:#bbbbbb;">              May 15, 2:37 PM PDT            </p>                                    <div class="zd-comment" dir="auto" style="color: #2b2e2f; font-family: 'Lucida Sans Unicode', 'Lucida Grande', 'Tahoma', Verdana, sans-serif; font-size: 14px; line-height: 22px; margin: 15px 0">Hi TJ,<br /><br />I think your attachment may not have sent correctly, we didn't receive anything – can you try again? Thanks!<br /><div class="signature"><p dir="auto" style="color: #2b2e2f; font-family: 'Lucida Sans Unicode', 'Lucida Grande', 'Tahoma', Verdana, sans-serif; font-size: 14px; line-height: 22px; margin: 15px 0">Best,<br />Jonathan</p><p dir="auto" style="color: #2b2e2f; font-family: 'Lucida Sans Unicode', 'Lucida Grande', 'Tahoma', Verdana, sans-serif; font-size: 14px; line-height: 22px; margin: 15px 0">Be sure to follow @Edison_Apps on <a href="http://www.twitter.com/edison_apps" rel="noreferrer">Twitter</a> and <a href="http://www.instagram.com/edison_apps" rel="noreferrer">Instagram</a>!</p></div></div><p>                                  </p></td>        </tr>      </table>    </td>  </tr></table><p></p><table width="100%" cellpadding="0" cellspacing="0" border="0" role="presentation">  <tr>    <td width="100%" style="padding: 15px 0; border-top: 1px dotted #c5c5c5;">      <table width="100%" cellpadding="0" cellspacing="0" border="0" style="table-layout:fixed;" role="presentation">        <tr>                    <td width="100%" style="padding: 0; margin: 0;" valign="top">            <p style="font-family: 'Lucida Grande','Lucida Sans Unicode','Lucida Sans',Verdana,Tahoma,sans-serif; font-size: 15px; line-height: 18px; margin-bottom: 0; margin-top: 0; padding: 0; color:#1b1d1e;">                              <strong>Tjliston2</strong>                          </p>            <p style="font-family: 'Lucida Grande','Lucida Sans Unicode','Lucida Sans',Verdana,Tahoma,sans-serif; font-size: 13px; line-height: 25px; margin-bottom: 15px; margin-top: 0; padding: 0; color:#bbbbbb;">              May 15, 7:43 AM PDT            </p>                                    <div class="zd-comment zd-comment-pre-styled" dir="auto"><div><div></div>Here is a video of the shaking and jumping around and losing your place when trying to expand a picture.  It happens on all emails </div><div>Thanks </div></div><p>                                  </p></td>        </tr>      </table>    </td>  </tr></table><p></p><table width="100%" cellpadding="0" cellspacing="0" border="0" role="presentation">  <tr>    <td width="100%" style="padding: 15px 0; border-top: 1px dotted #c5c5c5;">      <table width="100%" cellpadding="0" cellspacing="0" border="0" style="table-layout:fixed;" role="presentation">        <tr>                    <td width="100%" style="padding: 0; margin: 0;" valign="top">            <p style="font-family: 'Lucida Grande','Lucida Sans Unicode','Lucida Sans',Verdana,Tahoma,sans-serif; font-size: 15px; line-height: 18px; margin-bottom: 0; margin-top: 0; padding: 0; color:#1b1d1e;">                                                                    <strong>Jonathan</strong> (Edison Mail)                                                            </p>            <p style="font-family: 'Lucida Grande','Lucida Sans Unicode','Lucida Sans',Verdana,Tahoma,sans-serif; font-size: 13px; line-height: 25px; margin-bottom: 15px; margin-top: 0; padding: 0; color:#bbbbbb;">              May 9, 7:24 PM PDT            </p>                                    <div class="zd-comment" dir="auto" style="color: #2b2e2f; font-family: 'Lucida Sans Unicode', 'Lucida Grande', 'Tahoma', Verdana, sans-serif; font-size: 14px; line-height: 22px; margin: 15px 0">Hello TJ,<br /><br />Thanks for messaging us! I'm sorry you're running into this issue where your messages are being marked as spam incorrectly. Spam messages are something that's handled <i>server-side</i> by your email provider, and unfortunately we don't have control over this. We'd recommend getting in touch with them to see if there's anything you can do to prevent these messages from being marked as spam before it hits your inbox.<br /><br />The image issue you've reported is something I believe we're aware of, but just to make sure with engineering, would you mind making and sending us a screen recording of it in action? Thanks!<br /><div class="signature"><p dir="auto" style="color: #2b2e2f; font-family: 'Lucida Sans Unicode', 'Lucida Grande', 'Tahoma', Verdana, sans-serif; font-size: 14px; line-height: 22px; margin: 15px 0">Best,<br />Jonathan</p><p dir="auto" style="color: #2b2e2f; font-family: 'Lucida Sans Unicode', 'Lucida Grande', 'Tahoma', Verdana, sans-serif; font-size: 14px; line-height: 22px; margin: 15px 0">Be sure to follow @Edison_Apps on <a href="http://www.twitter.com/edison_apps" rel="noreferrer">Twitter</a> and <a href="http://www.instagram.com/edison_apps" rel="noreferrer">Instagram</a>!</p></div></div><p>                                  </p></td>        </tr>      </table>    </td>  </tr></table><p></p><table width="100%" cellpadding="0" cellspacing="0" border="0" role="presentation">  <tr>    <td width="100%" style="padding: 15px 0; border-top: 1px dotted #c5c5c5;">      <table width="100%" cellpadding="0" cellspacing="0" border="0" style="table-layout:fixed;" role="presentation">        <tr>                    <td width="100%" style="padding: 0; margin: 0;" valign="top">            <p style="font-family: 'Lucida Grande','Lucida Sans Unicode','Lucida Sans',Verdana,Tahoma,sans-serif; font-size: 15px; line-height: 18px; margin-bottom: 0; margin-top: 0; padding: 0; color:#1b1d1e;">                              <strong>Tjliston2</strong>                          </p>            <p style="font-family: 'Lucida Grande','Lucida Sans Unicode','Lucida Sans',Verdana,Tahoma,sans-serif; font-size: 13px; line-height: 25px; margin-bottom: 15px; margin-top: 0; padding: 0; color:#bbbbbb;">              May 8, 5:50 AM PDT            </p>                                    <div class="zd-comment" dir="auto" style="color: #2b2e2f; font-family: 'Lucida Sans Unicode', 'Lucida Grande', 'Tahoma', Verdana, sans-serif; font-size: 14px; line-height: 22px; margin: 15px 0"><p dir="auto" style="color: #2b2e2f; font-family: 'Lucida Sans Unicode', 'Lucida Grande', 'Tahoma', Verdana, sans-serif; font-size: 14px; line-height: 22px; margin: 15px 0">I sometimes like to send an article to my email account for future reading but Edison is sending email from myself into my spam folder.  How can i keep certain addresses, including my own, from automatically going into the spam folder?<br />On another note when I am in Edison and try and expand a photo inside an email it wiggles and jumps around so that you can't see it.<br />thanks<br />TJ Liston<br />iphone X  ios 12.2</p></div>                                  </td>        </tr>      </table>    </td>  </tr></table></div>
<span style='color:#FFFFFF' aria-hidden='true'>[Z3647L-5EVW]</span></body>
"""
html2= """<body>
    <div style="color: #b5b5b5;">##- Please type your reply above this line -##</div>
    <p></p>
    <div style="margin-top: 25px" data-version="2">
        <table width="100%" cellpadding="0" cellspacing="0" border="0" role="presentation">
            <tr>
                <td width="100%" style="padding: 15px 0; border-top: 1px dotted #c5c5c5;">
                    <table width="100%" cellpadding="0" cellspacing="0" border="0" style="table-layout:fixed;" role="presentation">
                        <tr>
                            <td width="100%" style="padding: 0; margin: 0;" valign="top">
                                <p style="font-family: 'Lucida Grande','Lucida Sans Unicode','Lucida Sans',Verdana,Tahoma,sans-serif; font-size: 15px; line-height: 18px; margin-bottom: 0; margin-top: 0; padding: 0; color:#1b1d1e;"> <strong>Jonathan</strong> (Edison Mail) </p>
                                <p style="font-family: 'Lucida Grande','Lucida Sans Unicode','Lucida Sans',Verdana,Tahoma,sans-serif; font-size: 13px; line-height: 25px; margin-bottom: 15px; margin-top: 0; padding: 0; color:#bbbbbb;"> May 15, 2:37 PM PDT </p>
                                <div class="zd-comment" dir="auto" style="color: #2b2e2f; font-family: 'Lucida Sans Unicode', 'Lucida Grande', 'Tahoma', Verdana, sans-serif; font-size: 14px; line-height: 22px; margin: 15px 0">Hi TJ,
                                    <br />
                                    <br />I think your attachment may not have sent correctly, we didn't receive anything – can you try again? Thanks!
                                    <br />
                                    <div class="signature">
                                        <p dir="auto" style="color: #2b2e2f; font-family: 'Lucida Sans Unicode', 'Lucida Grande', 'Tahoma', Verdana, sans-serif; font-size: 14px; line-height: 22px; margin: 15px 0">Best,
                                            <br />Jonathan</p>
                                        <p dir="auto" style="color: #2b2e2f; font-family: 'Lucida Sans Unicode', 'Lucida Grande', 'Tahoma', Verdana, sans-serif; font-size: 14px; line-height: 22px; margin: 15px 0">Be sure to follow @Edison_Apps on <a href="http://www.twitter.com/edison_apps" rel="noreferrer">Twitter</a> and <a href="http://www.instagram.com/edison_apps" rel="noreferrer">Instagram</a>!</p>
                                    </div>
                                </div>
                                <p> </p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
        <p></p>
        <table width="100%" cellpadding="0" cellspacing="0" border="0" role="presentation">
            <tr>
                <td width="100%" style="padding: 15px 0; border-top: 1px dotted #c5c5c5;">
                    <table width="100%" cellpadding="0" cellspacing="0" border="0" style="table-layout:fixed;" role="presentation">
                        <tr>
                            <td width="100%" style="padding: 0; margin: 0;" valign="top">
                                <p style="font-family: 'Lucida Grande','Lucida Sans Unicode','Lucida Sans',Verdana,Tahoma,sans-serif; font-size: 15px; line-height: 18px; margin-bottom: 0; margin-top: 0; padding: 0; color:#1b1d1e;"> <strong>Tjliston2</strong> </p>
                                <p style="font-family: 'Lucida Grande','Lucida Sans Unicode','Lucida Sans',Verdana,Tahoma,sans-serif; font-size: 13px; line-height: 25px; margin-bottom: 15px; margin-top: 0; padding: 0; color:#bbbbbb;"> May 15, 7:43 AM PDT </p>
                                <div class="zd-comment zd-comment-pre-styled" dir="auto">
                                    <div>
                                        <div></div>Here is a video of the shaking and jumping around and losing your place when trying to expand a picture.  It happens on all emails </div>
                                    <div>Thanks </div>
                                </div>
                                <p> </p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
        <p></p>
        <table width="100%" cellpadding="0" cellspacing="0" border="0" role="presentation">
            <tr>
                <td width="100%" style="padding: 15px 0; border-top: 1px dotted #c5c5c5;">
                    <table width="100%" cellpadding="0" cellspacing="0" border="0" style="table-layout:fixed;" role="presentation">
                        <tr>
                            <td width="100%" style="padding: 0; margin: 0;" valign="top">
                                <p style="font-family: 'Lucida Grande','Lucida Sans Unicode','Lucida Sans',Verdana,Tahoma,sans-serif; font-size: 15px; line-height: 18px; margin-bottom: 0; margin-top: 0; padding: 0; color:#1b1d1e;"> <strong>Jonathan</strong> (Edison Mail) </p>
                                <p style="font-family: 'Lucida Grande','Lucida Sans Unicode','Lucida Sans',Verdana,Tahoma,sans-serif; font-size: 13px; line-height: 25px; margin-bottom: 15px; margin-top: 0; padding: 0; color:#bbbbbb;"> May 9, 7:24 PM PDT </p>
                                <div class="zd-comment" dir="auto" style="color: #2b2e2f; font-family: 'Lucida Sans Unicode', 'Lucida Grande', 'Tahoma', Verdana, sans-serif; font-size: 14px; line-height: 22px; margin: 15px 0">Hello TJ,
                                    <br />
                                    <br />Thanks for messaging us! I'm sorry you're running into this issue where your messages are being marked as spam incorrectly. Spam messages are something that's handled <i>server-side</i> by your email provider, and unfortunately we don't have control over this. We'd recommend getting in touch with them to see if there's anything you can do to prevent these messages from being marked as spam before it hits your inbox.
                                    <br />
                                    <br />The image issue you've reported is something I believe we're aware of, but just to make sure with engineering, would you mind making and sending us a screen recording of it in action? Thanks!
                                    <br />
                                    <div class="signature">
                                        <p dir="auto" style="color: #2b2e2f; font-family: 'Lucida Sans Unicode', 'Lucida Grande', 'Tahoma', Verdana, sans-serif; font-size: 14px; line-height: 22px; margin: 15px 0">Best,
                                            <br />Jonathan</p>
                                        <p dir="auto" style="color: #2b2e2f; font-family: 'Lucida Sans Unicode', 'Lucida Grande', 'Tahoma', Verdana, sans-serif; font-size: 14px; line-height: 22px; margin: 15px 0">Be sure to follow @Edison_Apps on <a href="http://www.twitter.com/edison_apps" rel="noreferrer">Twitter</a> and <a href="http://www.instagram.com/edison_apps" rel="noreferrer">Instagram</a>!</p>
                                    </div>
                                </div>
                                <p> </p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
        <p></p>
        <table width="100%" cellpadding="0" cellspacing="0" border="0" role="presentation">
            <tr>
                <td width="100%" style="padding: 15px 0; border-top: 1px dotted #c5c5c5;">
                    <table width="100%" cellpadding="0" cellspacing="0" border="0" style="table-layout:fixed;" role="presentation">
                        <tr>
                            <td width="100%" style="padding: 0; margin: 0;" valign="top">
                                <p style="font-family: 'Lucida Grande','Lucida Sans Unicode','Lucida Sans',Verdana,Tahoma,sans-serif; font-size: 15px; line-height: 18px; margin-bottom: 0; margin-top: 0; padding: 0; color:#1b1d1e;"> <strong>Tjliston2</strong> </p>
                                <p style="font-family: 'Lucida Grande','Lucida Sans Unicode','Lucida Sans',Verdana,Tahoma,sans-serif; font-size: 13px; line-height: 25px; margin-bottom: 15px; margin-top: 0; padding: 0; color:#bbbbbb;"> May 8, 5:50 AM PDT </p>
                                <div class="zd-comment" dir="auto" style="color: #2b2e2f; font-family: 'Lucida Sans Unicode', 'Lucida Grande', 'Tahoma', Verdana, sans-serif; font-size: 14px; line-height: 22px; margin: 15px 0">
                                    <p dir="auto" style="color: #2b2e2f; font-family: 'Lucida Sans Unicode', 'Lucida Grande', 'Tahoma', Verdana, sans-serif; font-size: 14px; line-height: 22px; margin: 15px 0">I sometimes like to send an article to my email account for future reading but Edison is sending email from myself into my spam folder. How can i keep certain addresses, including my own, from automatically going into the spam folder?
                                        <br />On another note when I am in Edison and try and expand a photo inside an email it wiggles and jumps around so that you can't see it.
                                        <br />thanks
                                        <br />TJ Liston
                                        <br />iphone X ios 12.2</p>
                                </div>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
    </div>
    <span style='color:#FFFFFF' aria-hidden='true'>[Z3647L-5EVW]</span></body>"""

# print(html2text.html2text(html))
# <html>

soup = BeautifulSoup("<p>Some<b>bad<i>HTML", 'html.parser')

def get_sig(file):
    soup2 = BeautifulSoup(file, 'html.parser')
    # print(soup2.table.tr.td.table.tr.td.div['class'])
    # for tablet in soup2.table:
    #     print(type(tablet))
    #     body = tablet.tr.td.table.tr.td.div.strings
    ##recur tables

    body = soup2.table.tr.td.table.tr.td.div.p.strings
    doc = []
    text=""
    for line in body:
        line = line.replace("  ","")
        doc.append(line)
        text+= line
        # print(repr(line))
    print(text)
    return doc

# print(doc)
if __name__ == '__main__':
    get_sig(html2)

# print(soup2.find_alid="link3"l('div'))
# print(soup2.prettify())
# print(soup.find(text="bad"))
# <body>
# <div style="color: #b5b5b5;">##- Please type your reply above this line -##</div>
# <p></p><div style=" margin-top: 25px" data-version="2"><table width="100%" cellpadding="0" cellspacing="0" border="0" role="presentation">  <tr>    <td width="100%" style="padding: 15px 0; border-top: 1px dotted #c5c5c5;">      <table width="100%" cellpadding="0" cellspacing="0" border="0" style="table-layout:fixed;" role="presentation">        <tr>                    <td width="100%" style="padding: 0; margin: 0;" valign="top">            <p style="font-family: 'Lucida Grande','Lucida Sans Unicode','Lucida Sans',Verdana,Tahoma,sans-serif; font-size: 15px; line-height: 18px; margin-bottom: 0; margin-top: 0; padding: 0; color:#1b1d1e;">                                                                    <strong>Jonathan</strong> (Edison Mail)                                                            </p>            <p style="font-family: 'Lucida Grande','Lucida Sans Unicode','Lucida Sans',Verdana,Tahoma,sans-serif; font-size: 13px; line-height: 25px; margin-bottom: 15px; margin-top: 0; padding: 0; color:#bbbbbb;">              May 15, 2:37 PM PDT            </p>                                    <div class="zd-comment" dir="auto" style="color: #2b2e2f; font-family: 'Lucida Sans Unicode', 'Lucida Grande', 'Tahoma', Verdana, sans-serif; font-size: 14px; line-height: 22px; margin: 15px 0">Hi TJ,<br /><br />I think your attachment may not have sent correctly, we didn't receive anything – can you try again? Thanks!<br /><div class="signature"><p dir="auto" style="color: #2b2e2f; font-family: 'Lucida Sans Unicode', 'Lucida Grande', 'Tahoma', Verdana, sans-serif; font-size: 14px; line-height: 22px; margin: 15px 0">Best,<br />Jonathan</p><p dir="auto" style="color: #2b2e2f; font-family: 'Lucida Sans Unicode', 'Lucida Grande', 'Tahoma', Verdana, sans-serif; font-size: 14px; line-height: 22px; margin: 15px 0">Be sure to follow @Edison_Apps on <a href="http://www.twitter.com/edison_apps" rel="noreferrer">Twitter</a> and <a href="http://www.instagram.com/edison_apps" rel="noreferrer">Instagram</a>!</p></div></div><p>                                  </p></td>        </tr>      </table>    </td>  </tr></table><p></p><table width="100%" cellpadding="0" cellspacing="0" border="0" role="presentation">  <tr>    <td width="100%" style="padding: 15px 0; border-top: 1px dotted #c5c5c5;">      <table width="100%" cellpadding="0" cellspacing="0" border="0" style="table-layout:fixed;" role="presentation">        <tr>                    <td width="100%" style="padding: 0; margin: 0;" valign="top">            <p style="font-family: 'Lucida Grande','Lucida Sans Unicode','Lucida Sans',Verdana,Tahoma,sans-serif; font-size: 15px; line-height: 18px; margin-bottom: 0; margin-top: 0; padding: 0; color:#1b1d1e;">                              <strong>Tjliston2</strong>                          </p>            <p style="font-family: 'Lucida Grande','Lucida Sans Unicode','Lucida Sans',Verdana,Tahoma,sans-serif; font-size: 13px; line-height: 25px; margin-bottom: 15px; margin-top: 0; padding: 0; color:#bbbbbb;">              May 15, 7:43 AM PDT            </p>                                    <div class="zd-comment zd-comment-pre-styled" dir="auto"><div><div></div>Here is a video of the shaking and jumping around and losing your place when trying to expand a picture.  It happens on all emails </div><div>Thanks </div></div><p>                                  </p></td>        </tr>      </table>    </td>  </tr></table><p></p><table width="100%" cellpadding="0" cellspacing="0" border="0" role="presentation">  <tr>    <td width="100%" style="padding: 15px 0; border-top: 1px dotted #c5c5c5;">      <table width="100%" cellpadding="0" cellspacing="0" border="0" style="table-layout:fixed;" role="presentation">        <tr>                    <td width="100%" style="padding: 0; margin: 0;" valign="top">            <p style="font-family: 'Lucida Grande','Lucida Sans Unicode','Lucida Sans',Verdana,Tahoma,sans-serif; font-size: 15px; line-height: 18px; margin-bottom: 0; margin-top: 0; padding: 0; color:#1b1d1e;">                                                                    <strong>Jonathan</strong> (Edison Mail)                                                            </p>            <p style="font-family: 'Lucida Grande','Lucida Sans Unicode','Lucida Sans',Verdana,Tahoma,sans-serif; font-size: 13px; line-height: 25px; margin-bottom: 15px; margin-top: 0; padding: 0; color:#bbbbbb;">              May 9, 7:24 PM PDT            </p>                                    <div class="zd-comment" dir="auto" style="color: #2b2e2f; font-family: 'Lucida Sans Unicode', 'Lucida Grande', 'Tahoma', Verdana, sans-serif; font-size: 14px; line-height: 22px; margin: 15px 0">Hello TJ,<br /><br />Thanks for messaging us! I'm sorry you're running into this issue where your messages are being marked as spam incorrectly. Spam messages are something that's handled <i>server-side</i> by your email provider, and unfortunately we don't have control over this. We'd recommend getting in touch with them to see if there's anything you can do to prevent these messages from being marked as spam before it hits your inbox.<br /><br />The image issue you've reported is something I believe we're aware of, but just to make sure with engineering, would you mind making and sending us a screen recording of it in action? Thanks!<br /><div class="signature"><p dir="auto" style="color: #2b2e2f; font-family: 'Lucida Sans Unicode', 'Lucida Grande', 'Tahoma', Verdana, sans-serif; font-size: 14px; line-height: 22px; margin: 15px 0">Best,<br />Jonathan</p><p dir="auto" style="color: #2b2e2f; font-family: 'Lucida Sans Unicode', 'Lucida Grande', 'Tahoma', Verdana, sans-serif; font-size: 14px; line-height: 22px; margin: 15px 0">Be sure to follow @Edison_Apps on <a href="http://www.twitter.com/edison_apps" rel="noreferrer">Twitter</a> and <a href="http://www.instagram.com/edison_apps" rel="noreferrer">Instagram</a>!</p></div></div><p>                                  </p></td>        </tr>      </table>    </td>  </tr></table><p></p><table width="100%" cellpadding="0" cellspacing="0" border="0" role="presentation">  <tr>    <td width="100%" style="padding: 15px 0; border-top: 1px dotted #c5c5c5;">      <table width="100%" cellpadding="0" cellspacing="0" border="0" style="table-layout:fixed;" role="presentation">        <tr>                    <td width="100%" style="padding: 0; margin: 0;" valign="top">            <p style="font-family: 'Lucida Grande','Lucida Sans Unicode','Lucida Sans',Verdana,Tahoma,sans-serif; font-size: 15px; line-height: 18px; margin-bottom: 0; margin-top: 0; padding: 0; color:#1b1d1e;">                              <strong>Tjliston2</strong>                          </p>            <p style="font-family: 'Lucida Grande','Lucida Sans Unicode','Lucida Sans',Verdana,Tahoma,sans-serif; font-size: 13px; line-height: 25px; margin-bottom: 15px; margin-top: 0; padding: 0; color:#bbbbbb;">              May 8, 5:50 AM PDT            </p>                                    <div class="zd-comment" dir="auto" style="color: #2b2e2f; font-family: 'Lucida Sans Unicode', 'Lucida Grande', 'Tahoma', Verdana, sans-serif; font-size: 14px; line-height: 22px; margin: 15px 0"><p dir="auto" style="color: #2b2e2f; font-family: 'Lucida Sans Unicode', 'Lucida Grande', 'Tahoma', Verdana, sans-serif; font-size: 14px; line-height: 22px; margin: 15px 0">I sometimes like to send an article to my email account for future reading but Edison is sending email from myself into my spam folder.  How can i keep certain addresses, including my own, from automatically going into the spam folder?<br />On another note when I am in Edison and try and expand a photo inside an email it wiggles and jumps around so that you can't see it.<br />thanks<br />TJ Liston<br />iphone X  ios 12.2</p></div>                                  </td>        </tr>      </table>    </td>  </tr></table></div>
# <span style='color:#FFFFFF' aria-hidden='true'>[Z3647L-5EVW]</span></body>
# </html>
